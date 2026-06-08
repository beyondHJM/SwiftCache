import os
import json
from typing import Optional

def init_vl_processor(model_path: str) -> Optional[object]:
    """
    Initialize vision-language processor if model is a VLM.
    Returns None if not a VLM or initialization fails.
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        return None

    # Locate config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
        # Load model config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        # print('model_type',model_type)
        # Define known VLM model type keywords
        # VLM_TYPES = {
        #     "deepseek_vl",
        #     "llava",
        #     "qwen_vl",
        #     "cogvlm",
        #     "internvl",
        #     "minigpt4",
        #     "blip2",
        #     "instructblip",
        #     "idefics",
        #     "bunny",
        #     "phi3_v",
        #     "xcomposer",
        #     "visualglm",
        #     "emu2",
        #     "sharegpt4v",
        #     "kosmos",
        # }

            # First, try loading with AutoProcessor (generic)
        if model_type == "deepseek_vl_v2":
            from swiftcache.models.deepseek import DeepseekVLV2Processor
            processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
            return processor
        else:
            # Not a VLM
            return None

    except Exception as e:
        print(f"Failed to initialize VL processor: {e}")
        return None


# Usage Example
if __name__ == '__main__':
    model_path = "/home/admin/workspace/aop_lab/app_data/.cache/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa"
    vl_chat_processor = init_vl_processor(model_path)

    if vl_chat_processor is not None:
        print("Vision-Language Processor is ready.")
    else:
        print("No Vision-Language Processor loaded (not a VLM or failed to load).")