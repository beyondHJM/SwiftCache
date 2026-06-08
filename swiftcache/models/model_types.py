# model_types.py 或直接放在同一文件顶部
from enum import Enum

class ModelType(Enum):
    LANGUAGE_ONLY = "language_only"
    VISION_LANGUAGE = "vision_language"
    # AUDIO_LANGUAGE = "audio_language"
    # MULTIMODAL = "multimodal"