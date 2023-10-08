from abc import ABC, abstractmethod
import numpy as np


class IColorConverter(ABC):
    @abstractmethod
    def convert(self, image):
        pass


class RgbToYcbcrConverter(IColorConverter):
    cvt_matrix = np.array([[0.299, 0.587, 0.114],
                           [-0.1687, -0.3313, 0.5],
                           [0.5, -0.4187, -0.0813]]).transpose()

    def convert(self, image):
        ycbcr = image.astype(np.float64).dot(self.cvt_matrix)
        ycbcr[:, :, 1] += 128  # add 128 to Cb channel
        ycbcr[:, :, 2] += 128  # add 128 to Cr channel

        return ycbcr.astype(np.uint8)


class YcbcrToRgbConverter(IColorConverter):
    cvt_matrix = np.array([[1, 0, 1.402],
                           [1, -0.34414, -0.71414],
                           [1, 1.772, 0]]).transpose()

    def convert(self, image):
        ycbcr = image.astype(np.float64)
        ycbcr[:, :, 1] -= 128  # subtract 128 from Cb channel
        ycbcr[:, :, 2] -= 128  # subtract 128 from Cr channel

        rgb = ycbcr.dot(self.cvt_matrix)
        np.clip(rgb, 0, 255, out=rgb)

        return rgb.astype(np.uint8)
