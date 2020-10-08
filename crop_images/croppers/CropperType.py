from enum import Enum, auto


class CropperType(Enum):
    CENTERED_CROPPER = auto()
    YOLO_CROPPER = auto()
