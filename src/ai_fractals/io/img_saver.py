# img_saver.py

from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt


class ImageSaver(ABC):
    @abstractmethod
    def save(self, path, img):
        pass


class PltImageSaver(ImageSaver):
    def __init__(self, cmap="twilight_shifted"):
        self.cmap = cmap

    def save(self, path, img):
        plt.imsave(path, img, cmap=self.cmap)


class OpenCVImageSaver(ImageSaver):
    def __init__(self, convert_rgb=True):
        self.convert_rgb = convert_rgb

    def save(self, path, img):
        if self.convert_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
