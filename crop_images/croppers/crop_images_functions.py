from typing import List, Tuple
import cv2

from .AbstractCropper import AbstractCropper
from .CropperType import CropperType
from .CenteredCropper import CenteredCropper
from .YOLOCropper import YOLOCropper


def cropper_factory(cropper_type: CropperType) -> AbstractCropper:
    if cropper_type == CropperType.CENTERED_CROPPER:
        return CenteredCropper()
    elif cropper_type == CropperType.YOLO_CROPPER:
        return YOLOCropper()
    else:
        raise NotImplementedError("Not a valid cropper")


def crop_images(images_paths: List[str], relation: Tuple = (4, 6), cropper_type: CropperType = CropperType.CENTERED_CROPPER):
    return [cropper_factory(cropper_type).crop(cv2.imread(file), relation) for file in images_paths]
