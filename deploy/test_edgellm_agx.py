import argparse
import torch
import time
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from ctypes import cdll
from llm_lib.edgellm_sampler import Sampler
import pickle


class InferTrtLLM(object):
    def __init__(self, logger, model_config, token_emb, stream, engine_path, opt_seq_len, rotary_emb,
                plugin_path, temperature, top_p, top_k,
                onnx_path=None, 
                kv_cache_capacity=4096):
        self.kv_cache_capacity = kv_cache_capacity
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.token_emb = token_emb
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
        
        cos, sin = rotary_emb(torch.zeros(1, dtype=torch.float16).cuda())
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
        while selected_token_id != self.model_config.eos_token_id and len(output_ids) <= max_new_tokens:
            # decode
            selected_token_id = self.decode(self.hidden_states)
            self.hidden_states = self.embed_tokens(
                torch.tensor([[selected_token_id]], device="cuda:0", dtype=torch.long)
            )
            output_ids.append(selected_token_id)
        return torch.tensor(output_ids, device="cuda:0", dtype=torch.long).unsqueeze(0)

class InferTrt(object):
    def __init__(self, logger, stream, LLM_engine=None):        
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
        self.LLM_engine = LLM_engine

    
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

    def __call__(self, img, img2lidars, command, can_bus, is_first_frame, ego_pose, ego_pose_inv, timestamp, input_ids):
        return self.forward(img=torch.from_numpy(img).to(device="cuda:0").contiguous(), 
                            img2lidars=torch.from_numpy(img2lidars).to(device="cuda:0").contiguous(), 
                            command=torch.from_numpy(command).to(device="cuda:0").contiguous(), 
                            can_bus=torch.from_numpy(can_bus).to(device="cuda:0").contiguous(), 
                            is_first_frame=torch.from_numpy(is_first_frame).to(device="cuda:0").contiguous(), 
                            ego_pose=torch.from_numpy(ego_pose).to(device="cuda:0").contiguous(), 
                            ego_pose_inv=torch.from_numpy(ego_pose_inv).to(device="cuda:0").contiguous(), 
                            timestamp=torch.from_numpy(timestamp).to(device="cuda:0").contiguous(),
                            input_ids=input_ids)

    def forward(self, img, img2lidars, command, can_bus, is_first_frame, ego_pose, ego_pose_inv, timestamp, input_ids):
        if len(self.bindings) == 0:
            print("Need to call eval() before forward!.")
            exit(-1)
        self.bindings["img"].copy_(img)
        self.bindings["img2lidars"].copy_(img2lidars)
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
        return output_ids_lst

def parse_args():
    parser = argparse.ArgumentParser(description="Test the EdgeLLM engine on Drive-AGX boards")
    parser.add_argument("--vision_engine_pth", type=str, required=True)
    parser.add_argument("--llm_engine_pth", type=str, required=True)
    parser.add_argument("--llm_onnx_path", type=str, required=True)
    parser.add_argument("--plugin_path", type=str, required=True)
    parser.add_argument("--load_input_path", type=str, required=True, help="Path to dumped input data")
    parser.add_argument("--save_output_path", type=str, required=True, help="Path to save output data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.load_input_path, "rb") as f:
        dump_data = pickle.load(f)
    cfg_dict = dump_data["model_config"]

    class MC:
        pass
    mc = MC()
    mc.hidden_size = cfg_dict["hidden_size"]
    mc.num_attention_heads = cfg_dict["num_attention_heads"]
    mc.num_hidden_layers = cfg_dict["num_hidden_layers"]
    mc.vocab_size = cfg_dict["vocab_size"]
    mc.num_key_value_heads = cfg_dict.get("num_key_value_heads")
    mc.pad_token_id = cfg_dict.get("pad_token_id")
    mc.eos_token_id = cfg_dict.get("eos_token_id")

    logger = trt.Logger(trt.Logger.WARNING)
    cuda_stream = torch.cuda.Stream()
    engine = InferTrt(logger, cuda_stream)
    engine.read(args.vision_engine_pth)

    token_emb_weight = torch.from_numpy(dump_data["token_emb"]).to(device="cuda:0").contiguous()
    rope = torch.from_numpy(dump_data["rope_rotary_cos_sin"]).to(device="cuda:0", dtype=torch.float32).contiguous()

    class DummyRotary:
        def __init__(self, rope_concat):
            self.rope_concat = rope_concat
        def __call__(self, x):
            head_dim = self.rope_concat.shape[-1]
            half = head_dim // 2
            cos = self.rope_concat[..., :half][None, ...].to(dtype=torch.float16, device=x.device)
            sin = self.rope_concat[..., half:][None, ...].to(dtype=torch.float16, device=x.device)
            return cos, sin
    rotary_emb = DummyRotary(rope)

    engine.LLM_engine = InferTrtLLM(logger, model_config=mc,
                                    stream=cuda_stream,
                                    engine_path=args.llm_engine_pth,
                                    opt_seq_len=577,
                                    rotary_emb=rotary_emb,
                                    onnx_path=args.llm_onnx_path,
                                    plugin_path=args.plugin_path,
                                    token_emb=token_emb_weight,
                                    temperature=0.1, top_p=0.75, top_k=0,
                                    kv_cache_capacity=4096)

    engine.eval()
    result_list = []
    for idx, sample in enumerate(dump_data["samples"]):
        output_ids_lst = engine(img=sample["img"], 
                                img2lidars=sample["img2lidars"],
                                command=sample["command"], 
                                can_bus=sample["can_bus"], 
                                is_first_frame=sample["is_first_frame"], 
                                ego_pose=sample["ego_pose"], 
                                ego_pose_inv=sample["ego_pose_inv"], 
                                timestamp=sample["timestamp"], 
                                input_ids=sample["input_ids"])
        output_ids_lst = [output_ids.detach().cpu().numpy()for output_ids in output_ids_lst]
        result_list.append({
            "output_ids_lst": output_ids_lst,
            "ref_output_ids": sample["ref_output_ids"],
        })
    with open(args.save_output_path, "wb") as f:
        pickle.dump(result_list, f)
    print(f"\nSaved {len(result_list)} LLM comparisons to {args.save_output_path}")

if __name__ == '__main__':
    cuda.init()
    torch.cuda.init()
    main()
