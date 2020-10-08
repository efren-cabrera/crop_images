from typing import Tuple
import numpy as np

from .AbstractCropper import AbstractCropper


class CenteredCropper(AbstractCropper):
    def crop(self, image, relation: Tuple):
        height, width, _ = image.shape
        crop_width, crop_height = self.find_crop_width_height(width, height, relation)

        x_0 = width/2 - (crop_width/2)
        x_f = width/2 + (crop_width/2)

        x_0, x_f = self.adjust_values(x_0, x_f, 0, width)

        y_0 = height/2 - (crop_height/2)
        y_f = height/2 + (crop_height/2)
        y_0, y_f = self.adjust_values(y_0, y_f, 0, height)

        return self.make_crop(image, x_0, x_f, y_0, y_f)



