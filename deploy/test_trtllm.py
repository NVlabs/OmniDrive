import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.apis import set_random_seed
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import NMSFreeCoder
from mmdet.datasets import replace_ImageToTensor
import time
import tensorrt as trt
import pycuda.driver as cuda
import os.path as osp
import numpy as np
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer
import json


class InferTrtLLM(object):
    def __init__(self, llm_engine_pth, tokenizer_pth) -> None:
        device_id = 0
        self.IMAGE_TOKEN_INDEX = -200
        self.llm_engine_pth = llm_engine_pth
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, model_max_length=2048, padding_side="right", use_fast=False,)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model_type = "llava_llama"
        self.init_llm()
    
    def init_llm(self):
        self.model = ModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=False, stream=self.stream)
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def image_to_ptuning(self, input_ids, vision_embeded):
        updated_input_ids = []
        current_vocab_size = self.tokenizer.vocab_size
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                updated_input_ids.append(cur_input_ids)
                continue
            im_token_ids = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist()
            im_token_ids = [-1] + im_token_ids + [cur_input_ids.shape[0]]
            im_idx = 0
            for i in range(len(im_token_ids) - 1):
                updated_input_ids.append(cur_input_ids[im_token_ids[i]+1:im_token_ids[i+1]])
                if im_idx < vision_embeded.shape[0]:
                    im = vision_embeded[im_idx]
                    im_size = im.shape[0]
                    im_indices = torch.from_numpy(np.arange(current_vocab_size, current_vocab_size + im_size)).cuda()
                    updated_input_ids.append(im_indices)
                    im_idx += 1
        return torch.cat(updated_input_ids).unsqueeze(0), vision_embeded.reshape(1, -1, vision_embeded.shape[2])

    def generate(self, input_ids, vision_embeded):
        input_ids, prompt_table = self.image_to_ptuning(input_ids, vision_embeded)
        input_ids = input_ids.contiguous().to(dtype=torch.int32)
        prompt_table = prompt_table.cuda().contiguous().to(dtype=torch.float16)
        t_start = time.time()
        output_ids = self.model.generate(
            input_ids, 
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=320,
            use_cache=False)
        # print(f"Generation time: {time.time() - t_start}s.")
        output_ids = torch.masked_select(output_ids, output_ids.lt(self.tokenizer.vocab_size)).reshape([1, -1])
        self.stream.synchronize()
        return output_ids


class InferTrt(object):
    def __init__(self, logger, qa_save_path, LLM_engine=None, torch_ref_model=None):        
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        self.cuda_ctx.push()

        self.builder = trt.Builder(logger)
        self.logger = logger
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.opt = self.builder.create_optimization_profile()

        self.config = self.builder.create_builder_config()
        self.config.add_optimization_profile(self.opt)
        # self.config.max_workspace_size = 2 << 34
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # self.config.set_flag(trt.BuilderFlag.FP16)  # control this
        self.stream = cuda.Stream()
        self.cuda_ctx.pop()
        self.curr_scene_token = None
        self.start_timestamp = None
        self.bindings = {}
        self.bbox_coder = NMSFreeCoder(
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            voxel_size=[0.2, 0.2, 8],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=300,
            score_threshold=None,
            num_classes=10
        )
        self.LLM_engine = LLM_engine
        self.qa_save_path = qa_save_path
        self.torch_ref_model = torch_ref_model

    
    def from_onnx(self, onnx_mod):
        parser = trt.OnnxParser(self.network, self.logger)
        result = parser.parse(onnx_mod.SerializeToString())
        if not result:
            print("failed parsing onnx")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
        self.buf = self.builder.build_serialized_network(self.network, self.config)
        self._build_engine()
        
    def _build_engine(self):
        self.runtime = trt.Runtime(self.logger)        
        self.engine = self.runtime.deserialize_cuda_engine(self.buf)
        self.context = self.engine.create_execution_context()
        # self.context.profiler = CustomProfiler()
        self.names = []
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            self.names.append(self.engine.get_tensor_name(i))

    def write(self, path):
        with open(path, "wb") as fp:
            fp.write(self.buf)

    def read(self, path):
        print("[TensorRT INFO] Loading engine from: ", path)
        with open(path, "rb") as fp:
            self.buf = fp.read()
        self._build_engine()

    def get_bbox(self, all_cls_scores, all_bbox_preds, img_metas):
        bbox_results = []
        preds_dicts = self.bbox_coder.decode({
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict': None,
        })
        num_samples = len(preds_dicts)

        bbox_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            bbox_list.append([bboxes, scores, labels])

        for bboxes, scores, labels in bbox_list:
            bbox_results.append(bbox3d2result(bboxes, scores, labels))
        return bbox_results
    
    def get_lane(self, all_lane_cls_one2one, all_lane_preds_one2one, img_metas):
        cls_scores = all_lane_cls_one2one[-1]
        bbox_preds = all_lane_preds_one2one[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            assert len(cls_score) == len(bbox_pred)
            cls_score = cls_score.sigmoid()
            det_bboxes = bbox_pred
            for p in range(11):
                det_bboxes[..., 3 * p].clamp_(min=-51.2000, max=51.2000)
                det_bboxes[..., 3 * p + 1].clamp_(min=-51.2000, max=51.2000)
                
            # det_bboxes = self.control_points_to_lane_points(det_bboxes)
            det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)
            result_list.append([det_bboxes.cpu().numpy(), cls_score.cpu().numpy()])
        return result_list
    
    def eval(self):
        if self.torch_ref_model is not None:            
            self.torch_ref_model.eval()
        if len(self.bindings) == 0:
            create_bindings_tensor = True
        else:
            create_bindings_tensor = False
        n_io = self.engine.num_io_tensors
        metas_in = []
        metas_out = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = str(self.engine.get_tensor_shape(tname))
            tdtype = str(self.engine.get_tensor_dtype(tname))
            tmode = str(self.engine.get_tensor_mode(tname))
            m = f"{i}\t{tname}\t{tshape}\t{tdtype}"
            if "INPUT" in tmode:
                metas_in.append(m)
            elif "OUTPUT" in tmode:
                metas_out.append(m)
            else:
                assert False, f"Unrecognized tensor mode: {tname}: {tmode}."
            if create_bindings_tensor:
                self.bindings[tname] = torch.zeros(list(self.engine.get_tensor_shape(tname)), 
                                        dtype=torch.float32, 
                                        device="cuda:0").contiguous()
        print("##### Input Bindings: ")
        print("\n".join(metas_in))
        print("##### Output Bindings: ")
        print("\n".join(metas_out))
        return

    def __call__(self, img_metas, input_ids, img, lidar2img, intrinsics, extrinsics, timestamp, img_timestamp, 
                ego_pose, ego_pose_inv, command, can_bus,
                return_loss=False, rescale=True):
        if self.torch_ref_model is not None:
            data_dict = {
                "img_metas": img_metas, "input_ids": input_ids, "img": img, "lidar2img": lidar2img, 
                "intrinsics": intrinsics, "extrinsics": extrinsics, "timestamp": timestamp, "img_timestamp": img_timestamp, 
                "ego_pose": ego_pose, "ego_pose_inv": ego_pose_inv, "command": command, "can_bus": can_bus,
            }
            ref_result_list = self.torch_ref_model(return_loss=False, rescale=True, **data_dict)
        else:
            ref_result_list = None
        return self.forward(img_metas=img_metas, 
                            input_ids=input_ids, 
                            img=img, 
                            lidar2img=lidar2img, 
                            intrinsics=intrinsics, 
                            timestamp=timestamp, 
                            ego_pose=ego_pose, 
                            ego_pose_inv=ego_pose_inv, 
                            command=command, 
                            can_bus=can_bus)

    def forward(self, img_metas, input_ids, img, lidar2img, intrinsics, timestamp, 
                ego_pose, ego_pose_inv, command, can_bus):
        if len(self.bindings) == 0:
            print("Need to call eval() before forward!.")
            exit(-1)
        # re-format the data
        img_metas = img_metas[0].data[0]
        input_ids = input_ids[0].data[0]
        img = img[0].data[0].to(device="cuda:0").contiguous()
        img2lidar = lidar2img[0].data[0][0].unsqueeze(0).to(device="cuda:0").inverse()
        intrinsics = intrinsics[0].data[0][0].unsqueeze(0).to(device="cuda:0")
        timestamp = timestamp[0].data[0][0].unsqueeze(0)
        ego_pose = ego_pose[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        ego_pose_inv = ego_pose_inv[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        command = command[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        can_bus = can_bus[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        # convert timestamp from fp64 to fp32
        if self.curr_scene_token is None or img_metas[0]["scene_token"] != self.curr_scene_token:
            self.start_timestamp = timestamp[0].item()
            self.curr_scene_token = img_metas[0]["scene_token"]
            is_first_frame = torch.ones([1]).to(device="cuda:0").contiguous()
        else:
            is_first_frame = torch.zeros([1]).to(device="cuda:0").contiguous()
        timestamp -= self.start_timestamp
        timestamp = timestamp.type(torch.float32).to(device="cuda:0").contiguous()

        # copy the input values to the binding buffer
        self.bindings["img"].copy_(img)
        # self.bindings["intrinsics"].copy_(intrinsics.type(torch.float32).to(device="cuda:0").contiguous())
        self.bindings["img2lidars"].copy_(img2lidar.type(torch.float32).to(device="cuda:0").contiguous())
        self.bindings["command"].copy_(command)
        self.bindings["can_bus"].copy_(can_bus)
        self.bindings["is_first_frame"].copy_(is_first_frame)
        self.bindings["ego_pose"].copy_(ego_pose)
        self.bindings["ego_pose_inv"].copy_(ego_pose_inv)
        self.bindings["timestamp"].copy_(timestamp)
        self.bindings["memory_embedding_bbox_in"].copy_(self.bindings["memory_embedding_bbox_out"])
        self.bindings["memory_reference_point_bbox_in"].copy_(self.bindings["memory_reference_point_bbox_out"])
        self.bindings["memory_timestamp_bbox_in"].copy_(self.bindings["memory_timestamp_bbox_out"])
        self.bindings["memory_egopose_bbox_in"].copy_(self.bindings["memory_egopose_bbox_out"])
        self.bindings["memory_canbus_bbox_in"].copy_(self.bindings["memory_canbus_bbox_out"])
        self.bindings["sample_time_bbox_in"].copy_(self.bindings["sample_time_bbox_out"])
        self.bindings["memory_timestamp_map_in"].copy_(self.bindings["memory_timestamp_map_out"])
        self.bindings["sample_time_map_in"].copy_(self.bindings["sample_time_map_out"])
        self.bindings["memory_egopose_map_in"].copy_(self.bindings["memory_egopose_map_out"])
        self.bindings["memory_embedding_map_in"].copy_(self.bindings["memory_embedding_map_out"])
        self.bindings["memory_reference_point_map_in"].copy_(self.bindings["memory_reference_point_map_out"])
        # inference
        self.cuda_ctx.push()
        for i in range(len(self.names)):
            self.context.set_tensor_address(self.names[i], self.bindings[str(self.names[i])].data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        self.cuda_ctx.pop()

        output_ids_lst = []
        for q_id, input_llm_id in enumerate(input_ids[0]):
            input_llm_id = input_llm_id.unsqueeze(0).to(device="cuda:0").contiguous()
            output_ids = self.LLM_engine.generate(input_llm_id, self.bindings["vision_embeded"])
            output_ids_lst.append(output_ids)
        
        output_qa_lst = []
        for output_ids in output_ids_lst:
            output_text = self.LLM_engine.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_qa_lst.append(
                dict(
                    Q=img_metas[0]["vlm_labels"].data[q_id],
                    A=[qa_pair.split("\nYou")[1].split(". Please provide")[1].split(" ASSISTANT: ")[1] for qa_pair in output_text]
                )
            )
        with open(self.qa_save_path+img_metas[0]["sample_idx"], 'w') as qa_file:
            json.dump(output_qa_lst, qa_file)

        bbox_results = self.get_bbox(self.bindings["all_cls_scores"].clone(), 
                                     self.bindings["all_bbox_preds"].clone(), 
                                     img_metas)
        
        lane_results = self.get_lane(self.bindings["all_lane_cls_one2one"].clone(), 
                                     self.bindings["all_lane_preds_one2one"].clone(), 
                                     img_metas)

        result_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(result_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        result_list[0]['text_out'] = output_qa_lst
        result_list[0]['lane_results'] = lane_results
        return result_list

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) an engine')
    parser.add_argument('--config',help='test config file path')
    parser.add_argument('--engine_pth', help='engine file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    parser.add_argument('--llm_engine_pth', type=str, default=None)
    parser.add_argument('--tokenizer_pth', type=str, default=None)
    parser.add_argument('--qa_save_path', type=str, default=None)
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # # old versions did not save class info in checkpoints, this walkaround is
    # # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # # palette for visualization in segmentation tasks
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # elif hasattr(dataset, 'PALETTE'):
    #     # segmentation dataset has `PALETTE` attribute
    #     model.PALETTE = dataset.PALETTE
    
    if not os.path.exists(args.qa_save_path):
        os.makedirs(args.qa_save_path)
    # build the engine
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine = InferTrt(logger, args.qa_save_path)
    engine.read(args.engine_pth)
    # build LLM engine
    engine.LLM_engine = InferTrtLLM(llm_engine_pth=args.llm_engine_pth, tokenizer_pth=args.tokenizer_pth)

    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False)
        # engine.torch_ref_model = model
        outputs = custom_multi_gpu_test(engine, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    cuda.init()
    torch.cuda.init()
    main()
