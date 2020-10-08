from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AbstractCropper(ABC):
    @abstractmethod
    def crop(self, image, relation: Tuple):
        pass

    def adjust_values(self, n_0, n_f, limit_0, limit_f):
        if n_0 < limit_0:
            dif = limit_0 - n_0
            return np.floor(n_0 + dif), np.floor(n_f + dif)
        if n_f > limit_f:
            dif = n_f - limit_f
            return np.floor(n_0 - dif), np.floor(n_f - dif)
        return np.floor(n_0), np.floor(n_f)

    def find_crop_width_height(self, width, height, relation):
        crop_height = np.floor(width * relation[0] / relation[1])
        if crop_height > height:
            crop_height = height
            crop_width = np.floor(height * relation[1] / relation[0])
        else:
            crop_width = width
        return crop_width, crop_height

    def make_crop(self, image,  x_0, x_f, y_0, y_f):
        return image[int(y_0):int(y_f), int(x_0):int(x_f)]
