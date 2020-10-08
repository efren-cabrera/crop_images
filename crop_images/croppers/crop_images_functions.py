from typing import List, Tuple
import cv2

from .AbstractCropper import AbstractCropper
from .CropperType import CropperType
from .CenteredCropper import CenteredCropper


def centered_crop_images(images_paths: List[str], relation: Tuple = (4, 6)):
    return crop_images(images_paths, relation, cropper_factory(CropperType.CENTERED_CROPPER))


def cropper_factory(cropper_type: CropperType) -> AbstractCropper:
    if cropper_type == CropperType.CENTERED_CROPPER:
        return CenteredCropper()
    else:
        raise NotImplementedError("Not a valid cropper")


def crop_images(images_paths: List[str], relation: Tuple = (4, 6), cropper: AbstractCropper = CenteredCropper()):
    return [cropper.crop(cv2.imread(file), relation) for file in images_paths]
