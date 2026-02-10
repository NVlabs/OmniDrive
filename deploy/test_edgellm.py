import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
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
from ctypes import cdll
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_lib.edgellm_sampler import Sampler
import json
import pickle


class InferTrtLLM(object):
    def __init__(self, logger, model_config, token_emb, stream, engine_path, opt_seq_len, rotary_emb,
                plugin_path, tokenizer_path, temperature, top_p, top_k,
                onnx_path=None, 
                kv_cache_capacity=4096):
        self.kv_cache_capacity = kv_cache_capacity
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.token_emb = token_emb
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=2048, padding_side="right", use_fast=False,)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model_config = model_config
        assert(self.model_config.hidden_size % self.model_config.num_attention_heads == 0)
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        assert(head_dim % 2 == 0)
        self.stream = stream
        _ = cdll.LoadLibrary(plugin_path)
        runtime = trt.Runtime(logger)
        try:
            # read engine
            with open(engine_path, 'rb') as engine_file:
                serialized_engine = engine_file.read()
            print(f"Engine loaded from {engine_path}.")
        except:
            # build engine
            print("Building engine...")
            assert(onnx_path is not None)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(network, logger)
            parser.parse_from_file(onnx_path)
            builder_config = builder.create_builder_config()
            prefill_profile = builder.create_optimization_profile()
            prefill_profile.set_shape("context_lengths", [1,], [1,], [1,])
            prefill_profile.set_shape("rope_rotary_cos_sin", 
                                    [1, kv_cache_capacity, head_dim], 
                                    [1, kv_cache_capacity, head_dim], 
                                    [1, kv_cache_capacity, head_dim])
            prefill_profile.set_shape("inputs_embeds", 
                                    [1, 1, self.model_config.hidden_size], 
                                    [1, opt_seq_len, self.model_config.hidden_size], 
                                    [1, kv_cache_capacity, self.model_config.hidden_size])
            prefill_profile.set_shape("last_token_ids", [1,1], [1,1], [1,1])
            prefill_profile.set_shape("kvcache_start_index", [0,], [1,], [1,])

            for l in range(self.model_config.num_hidden_layers):
                prefill_profile.set_shape(f"past_key_values_{l}",
                                        [1, 2, self.model_config.num_key_value_heads, 0, head_dim], 
                                        [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim], 
                                        [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim])
            builder_config.add_optimization_profile(prefill_profile)

            decode_profile = builder.create_optimization_profile()   
            decode_profile.set_shape("context_lengths", [1,], [1,], [1,])
            decode_profile.set_shape("rope_rotary_cos_sin", 
                                    [1, kv_cache_capacity, head_dim], 
                                    [1, kv_cache_capacity, head_dim], 
                                    [1, kv_cache_capacity, head_dim])
            decode_profile.set_shape("inputs_embeds", 
                                    [1, 1, self.model_config.hidden_size], 
                                    [1, 1, self.model_config.hidden_size], 
                                    [1, 1, self.model_config.hidden_size])
            decode_profile.set_shape("last_token_ids", [1,1], [1,1], [1,1])
            decode_profile.set_shape("kvcache_start_index", [1,], [1,], [1,])
            for l in range(self.model_config.num_hidden_layers):
                decode_profile.set_shape(f"past_key_values_{l}", 
                                        [1, 2, self.model_config.num_key_value_heads, 0, head_dim], 
                                        [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim], 
                                        [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim])
            builder_config.add_optimization_profile(decode_profile)
            builder_config.builder_optimization_level = 3

            serialized_engine = builder.build_serialized_network(network, builder_config)
            with open(engine_path, "wb") as engine_file:
                engine_file.write(serialized_engine)
            print(f"Engine saved to {engine_path}.")
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        self.buffers = {}
        self.buffers["context_lengths"] = torch.tensor([1], dtype=torch.int32).cuda().contiguous()
        for l in range(self.model_config.num_hidden_layers):
            self.buffers[f"kv_cache.{l}"] = torch.zeros(
                (1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim),
                dtype=torch.float16).cuda().contiguous()
        
        cos, sin = rotary_emb(torch.zeros(1, dtype=torch.float16).cuda(), kv_cache_capacity)
        half_head_dim = head_dim // 2
        rope_embed = torch.cat((cos[..., :half_head_dim], sin[..., :half_head_dim]), dim=-1).squeeze(0)
        self.buffers["rope_rotary_cos_sin"] = rope_embed[:, :kv_cache_capacity, :].to(torch.float32).cuda().contiguous()
        self.buffers["last_token_ids"] = torch.tensor([[1]], dtype=torch.int64).cuda().contiguous()
        self.buffers["logits"] = torch.zeros([1, 1, self.model_config.vocab_size], dtype=torch.float32).cuda().contiguous()
        self.buffers["kvcache_start_index"] = torch.zeros([1], dtype=torch.int32).cuda().contiguous()
        self.dummy_tensor = torch.zeros([1], dtype=torch.int32).cuda().contiguous()
        
        self.prefill_ctx = self.engine.create_execution_context()
        self.prefill_ctx.set_optimization_profile_async(0, self.stream.cuda_stream)
        self.prefill_ctx.set_tensor_address("context_lengths", self.buffers["context_lengths"].data_ptr())
        self.prefill_ctx.set_input_shape("context_lengths", [1,])
        self.prefill_ctx.set_tensor_address("last_token_ids", self.buffers["last_token_ids"].data_ptr())
        self.prefill_ctx.set_input_shape("last_token_ids", [1, 1])
        self.prefill_ctx.set_tensor_address("rope_rotary_cos_sin", self.buffers["rope_rotary_cos_sin"].data_ptr())
        self.prefill_ctx.set_input_shape("rope_rotary_cos_sin", [1, kv_cache_capacity, head_dim])
        self.prefill_ctx.set_tensor_address("kvcache_start_index", self.dummy_tensor.data_ptr())
        self.prefill_ctx.set_input_shape("kvcache_start_index", [0,])
        for l in range(self.model_config.num_hidden_layers):
            self.prefill_ctx.set_tensor_address(f"past_key_values_{l}", self.buffers[f"kv_cache.{l}"].data_ptr())
            self.prefill_ctx.set_input_shape(f"past_key_values_{l}", [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim])
            self.prefill_ctx.set_tensor_address(f"present_key_values_{l}", self.buffers[f"kv_cache.{l}"].data_ptr())
        self.prefill_ctx.set_tensor_address("logits", self.buffers["logits"].data_ptr())
    
        self.decode_ctx = self.engine.create_execution_context()
        self.decode_ctx.set_optimization_profile_async(1, self.stream.cuda_stream)
        self.decode_ctx.set_tensor_address("context_lengths", self.buffers["context_lengths"].data_ptr())
        self.decode_ctx.set_input_shape("context_lengths", [1,])
        self.decode_ctx.set_tensor_address("last_token_ids", self.buffers["last_token_ids"].data_ptr())
        self.decode_ctx.set_input_shape("last_token_ids", [1, 1])
        self.decode_ctx.set_tensor_address("rope_rotary_cos_sin", self.buffers["rope_rotary_cos_sin"].data_ptr())
        self.decode_ctx.set_input_shape("rope_rotary_cos_sin", [1, kv_cache_capacity, head_dim])
        self.decode_ctx.set_tensor_address("kvcache_start_index", self.buffers["kvcache_start_index"].data_ptr())
        self.decode_ctx.set_input_shape("kvcache_start_index", [1,])
        for l in range(self.model_config.num_hidden_layers):
            self.decode_ctx.set_tensor_address(f"past_key_values_{l}", self.buffers[f"kv_cache.{l}"].data_ptr())
            self.decode_ctx.set_input_shape(f"past_key_values_{l}", [1, 2, self.model_config.num_key_value_heads, kv_cache_capacity, head_dim])
            self.decode_ctx.set_tensor_address(f"present_key_values_{l}", self.buffers[f"kv_cache.{l}"].data_ptr())
        self.decode_ctx.set_tensor_address("logits", self.buffers["logits"].data_ptr())
        self.decode_ctx.set_input_shape("inputs_embeds", [1, 1, self.model_config.hidden_size])

        self.hidden_states = None
        print("Engine initialization done.")

    def __str__(self):
        n_io = self.engine.num_io_tensors
        metas = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = str(self.engine.get_tensor_shape(tname))
            tdtype = str(self.engine.get_tensor_dtype(tname))
            m = f"{tname} {tshape} {tdtype}"
            metas.append(m)
        ret = "\n".join(metas)
        return ret
    
    def embed_tokens(self, input_ids):
        return torch.nn.functional.embedding(
            input_ids, self.token_emb,
            padding_idx=getattr(self.model_config, 'pad_token_id', None)
        )
    
    def image_to_ptuning(self, input_ids, vision_embeded):
        IMAGE_TOKEN_INDEX = -200
        
        # Match llava_arch.py logic: reshape image features to [num_images, num_tokens, hidden_size]
        vision_embeded = vision_embeded.reshape(vision_embeded.shape[0], -1, vision_embeded.shape[-1])

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_input_embeds = self.embed_tokens(cur_input_ids.unsqueeze(0)).squeeze(0)
                cur_input_embeds = cur_input_embeds.to(dtype=vision_embeded.dtype, device=vision_embeded.device)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim).unsqueeze(0)).squeeze(0)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = vision_embeded[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features.to(dtype=cur_input_embeds.dtype))
            
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            new_input_embeds.append(cur_new_input_embeds)

        return torch.stack(new_input_embeds, dim=0)
      
    def prefill(self, inputs_embeds):
        assert(inputs_embeds.dtype == torch.float16)
        for l in range(self.model_config.num_hidden_layers):
            self.buffers[f"kv_cache.{l}"] *= 0
        seq_len = inputs_embeds.shape[1]
        self.buffers["context_lengths"][0] = seq_len
        self.buffers["last_token_ids"][0][0] = seq_len - 1
        inputs_embeds = inputs_embeds.cuda().contiguous()
        self.prefill_ctx.set_input_shape("inputs_embeds", [1, seq_len, self.model_config.hidden_size])
        self.prefill_ctx.set_tensor_address("inputs_embeds", inputs_embeds.data_ptr())

        self.stream.synchronize()
        start_time = time.time()
        self.prefill_ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        print(f"Prefill:{1000*(time.time() - start_time)}ms", end=";")

        self.buffers["last_token_ids"][0][0] = 0
        self.buffers["kvcache_start_index"][0] = seq_len
        return Sampler.sample(
            self.buffers["logits"].squeeze(1),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        ).item()
    
    def decode(self, hidden_states):
        seq_len = self.buffers["kvcache_start_index"][0] + 1
        self.buffers["context_lengths"][0] = seq_len
        hidden_states = hidden_states.contiguous()
        self.decode_ctx.set_tensor_address("inputs_embeds", hidden_states.data_ptr())

        self.stream.synchronize()
        start_time = time.time()
        self.decode_ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        print(f"Decode:{1000*(time.time() - start_time)}ms", end=";")
        self.buffers["kvcache_start_index"][0] += 1
        return Sampler.sample(
            self.buffers["logits"].squeeze(1),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        ).item()
    
    def generate(self, input_ids, vision_embeded, max_new_tokens=320):
        output_ids = []
        inputs_embeds = self.image_to_ptuning(input_ids, vision_embeded)
        # prefill
        selected_token_id = self.prefill(inputs_embeds)
        self.hidden_states = self.embed_tokens(
            torch.tensor([[selected_token_id]], device="cuda:0", dtype=torch.long)
        )
        output_ids.append(selected_token_id)
        while selected_token_id != self.tokenizer.eos_token_id and len(output_ids) <= max_new_tokens:
            # decode
            selected_token_id = self.decode(self.hidden_states)
            self.hidden_states = self.embed_tokens(
                torch.tensor([[selected_token_id]], device="cuda:0", dtype=torch.long)
            )
            output_ids.append(selected_token_id)
        return torch.tensor(output_ids, device="cuda:0", dtype=torch.long).unsqueeze(0)

class InferTrt(object):
    def __init__(self, logger, stream, qa_save_path, LLM_engine=None, dump_number=-1, dump_path=None):        
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
        self.stream = stream
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
        self.dump_number = dump_number
        if self.dump_number >= 0:
            assert dump_path is not None, "dump_path is required when dump_number >= 0"
        self.dump_path = dump_path
        self.dump_data_list = [] if self.dump_number > 0 else None

    
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
        self.stream.synchronize()
        start_time = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        print(f"\nVision:{1000*(time.time() - start_time)}ms", end=";")
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
                    A=[qa_pair.strip() for qa_pair in output_text]
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

        if self.dump_number > 0:
            dumped_data = {}
            for io_name in ["img", "img2lidars", "command", "can_bus", "is_first_frame", "ego_pose", "ego_pose_inv", "timestamp"]:
                dumped_data[io_name] = self.bindings[io_name].clone().detach().cpu().numpy()
            dumped_data["input_ids"] = input_ids.copy()
            dumped_data["ref_output_ids"] = [output_ids.clone().detach().cpu().numpy() for output_ids in output_ids_lst]
            self.dump_data_list.append(dumped_data)
            self.dump_number -= 1
        if self.dump_number == 0:
            full_dump = {
                "model_config": self.LLM_engine.model_config.to_dict(),
                "token_emb": self.LLM_engine.token_emb.clone().detach().cpu().numpy(),
                "rope_rotary_cos_sin": self.LLM_engine.buffers["rope_rotary_cos_sin"].clone().detach().cpu().numpy(),
                "samples": self.dump_data_list
            }
            with open(self.dump_path, "wb") as f:
                pickle.dump(full_dump, f)
            print("Dumped samples to dumped_data.pkl.")
            self.dump_number -= 1

        return result_list

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) an engine')
    parser.add_argument('--config',help='test config file path')
    parser.add_argument('--engine_pth', help='engine file path')
    parser.add_argument('--llm_checkpoint', help='checkpoint file for LLM (The LLM weights must absorb the OmniDrive checkpoint)', type=str, default=None)
    parser.add_argument('--llm_engine_pth', type=str, default=None)
    parser.add_argument('--llm_onnx_path', type=str, default=None)
    parser.add_argument('--plugin_path', type=str, default=None)
    parser.add_argument('--tokenizer_pth', type=str, default=None)
    parser.add_argument('--qa_save_path', type=str, default=None)
    parser.add_argument('--dump_number', type=int, default=-1)
    parser.add_argument('--dump_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
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

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.llm_checkpoint, 
        torch_dtype=torch.float16, 
        device_map='cpu'
    ).model
    token_emb_weight = hf_model.embed_tokens.weight.cuda()
    rotary_emb = hf_model.layers[0].self_attn.rotary_emb.cuda()

    if not os.path.exists(args.qa_save_path):
        os.makedirs(args.qa_save_path)
    # build the engine
    logger = trt.Logger(trt.Logger.WARNING)
    cuda_stream = torch.cuda.Stream()
    engine = InferTrt(logger, cuda_stream, args.qa_save_path, dump_number=args.dump_number, dump_path=args.dump_path)
    engine.read(args.engine_pth)
    # build LLM engine
    engine.LLM_engine = InferTrtLLM(logger, model_config=hf_model.config, 
                                    stream=cuda_stream, 
                                    engine_path=args.llm_engine_pth, 
                                    opt_seq_len=577,
                                    rotary_emb=rotary_emb,
                                    onnx_path=args.llm_onnx_path,
                                    plugin_path=args.plugin_path,
                                    token_emb=token_emb_weight,
                                    tokenizer_path=args.tokenizer_pth,
                                    temperature=0.1, top_p=0.75, top_k=0,
                                    kv_cache_capacity=4096)

    assert distributed
    outputs = custom_multi_gpu_test(engine, data_loader, tmpdir=None, gpu_collect=False)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {}
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))

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
