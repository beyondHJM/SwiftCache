import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from swiftcache.models.model_types import ModelType
import time
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    print('正在加载模型权重...')
    t = time.time()
    if model.MODEL_TYPE is ModelType.LANGUAGE_ONLY:
        _load_model_for_language_only(model,path)
    if model.MODEL_TYPE is ModelType.VISION_LANGUAGE:
        _load_model_for_vision_language(model,path)
    print(f'模型权重加载完毕，耗时{(time.time()-t):.2f} s')
        
def _load_model_for_language_only(model: nn.Module, path: str):
    # if model.MODEL_TYPE is ModelType.LANGUAGE_ONLY
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 找文件
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    bin_files = glob(os.path.join(path, "*.bin"))

    if safetensor_files:
        # 优先使用 safetensors
        target_files = safetensor_files
        use_safetensors = True
    elif bin_files:
        # 如果没有 safetensors，则用 bin
        target_files = bin_files
        use_safetensors = False
    else:
        raise FileNotFoundError(f"路径 {path} 下没有 safetensors 或 bin 文件.")

    # 遍历目标文件并加载
    for file in target_files:
        if use_safetensors:
            # safetensors 加载逻辑
            with safe_open(file, framework="pt", device="cpu") as f:
                for weight_name in f.keys():
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        try:
                            param = model.get_parameter(weight_name)
                        except Exception:

                            continue
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
        else:
            # bin 加载逻辑
            state_dict = torch.load(file, map_location="cpu")
            for weight_name, tensor in state_dict.items():
                # if "tokens" in weight_name or "lm" in weight_name:
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
    
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except Exception:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)


def _load_model_for_vision_language(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    vision_state={}
    projector_state={}
    other_state={}
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if 'language' in weight_name:
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                            param = model.get_parameter(weight_name)
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, f.get_tensor(weight_name))

                elif 'vision' in weight_name:
                    vision_state[weight_name.replace("vision.",'')] = f.get_tensor(weight_name)
                
                elif 'projector' in weight_name:
                    projector_state[weight_name.replace("projector.",'')] = f.get_tensor(weight_name)
                
                else:
                    other_state[weight_name] = f.get_tensor(weight_name)

    # for name, param in model.vision.named_parameters():

    #     print(name,param.shape)
    model.vision.load_state_dict(vision_state,strict=True)
    model.projector.load_state_dict(projector_state,strict=True)