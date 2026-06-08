from enum import Enum
from typing import Optional, Any
from swiftcache.engine.utils.io import load_pil_images
class VisionStatus(Enum):
    PENDDING = "pendding"
    PROCESSING = "processing"
    PROJECTING = "projecting"
    COMPLETED = "completed"

class VisionSequenceData:
    """Encapsulates all vision-related state and data for a VLM sequence."""

    def __init__(self, raw_input: Any):

        assert isinstance(raw_input, dict), f"raw_input must be dict, got {type(raw_input).__name__}"
        self.raw_input = raw_input
        self.pil_images = load_pil_images(raw_input)
        self.status: VisionStatus = VisionStatus.PENDDING
        self.prepare_inputs: Optional[Any] = None 
        self.image_embeds: Optional[Any] = None      # Raw vision encoder output
        self.projected_embeds: Optional[Any] = None  # After MLP projector

    def update_status(self, new_status: VisionStatus):
        self.status = new_status

    def is_completed(self) -> bool:
        return self.status == VisionStatus.COMPLETED

    def __repr__(self):
        return (f"VisionData(status={self.status.value}, "
                f"has_image_embeds={self.image_embeds is not None}, "
                f"has_projected={self.projected_embeds is not None})")