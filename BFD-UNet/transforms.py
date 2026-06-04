import random
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # [C,H,W], float32, 自动 /255
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else min_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size  # (h, w) or int

    def __call__(self, image, target):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        target = F.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        return image, target


def pad_if_smaller(img, size, fill=0):
    # PIL.Image.size = (w, h)
    w, h = img.size
    if min(w, h) < size:
        padw = max(size - w, 0)
        padh = max(size - h, 0)
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size, fill=0)
        target = pad_if_smaller(target, self.size, fill=255)

        w, h = image.size
        th, tw = self.size, self.size

        if w == tw and h == th:
            return image, target

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        image = F.crop(image, i, j, th, tw)
        target = F.crop(target, i, j, th, tw)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size, fill=0)
        target = pad_if_smaller(target, self.size, fill=255)
        image = F.center_crop(image, [self.size, self.size])
        target = F.center_crop(target, [self.size, self.size])
        return image, target