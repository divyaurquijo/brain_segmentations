from functools import partial
from typing import Callable, List, Tuple

import albumentations
import numpy as np
from sklearn.externals._pilutil import bytescale
import skimage.transform
from skimage.transform import resize
import torchvision.transforms


def transforms_training():
    """Transformations for training"""
    transforms_training = ComposeDouble(
        [
            FunctionWrapperDouble(
                resize, input=True, target=False, output_shape=(16, 128, 128)
            ),
            FunctionWrapperDouble(
                resize,
                input=False,
                target=True,
                output_shape=(16, 128, 128),
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ),
            AlbuSeg3d(albumentations.HorizontalFlip(p=0.5)),
            AlbuSeg3d(albumentations.VerticalFlip(p=0.5)),
            AlbuSeg3d(albumentations.Rotate(p=0.5)),
            AlbuSeg3d(albumentations.RandomRotate90(p=0.5)),
            FunctionWrapperDouble(create_dense_target, input=False, target=True),
            FunctionWrapperDouble(np.expand_dims, axis=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )
    return transforms_training


def transforms_testing():
    """Transformations for testing"""
    transforms_testing = ComposeDouble(
        [
            FunctionWrapperDouble(
                resize, input=True, target=False, output_shape=(16, 128, 128)
            ),
            FunctionWrapperDouble(
                resize,
                input=False,
                target=True,
                output_shape=(16, 128, 128),
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ),
            FunctionWrapperDouble(create_dense_target, input=False, target=True),
            FunctionWrapperDouble(np.expand_dims, axis=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )
    return transforms_testing


def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(
        self,
        function: Callable,
        input: bool = True,
        target: bool = False,
        *args,
        **kwargs,
    ):

        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input:
            inp = self.function(inp)
        if self.target:
            tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self):
        return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class AlbuSeg3d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation, so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (spatial_dims)  -> No (C)hannel dimension
    Expected target: (spatial_dims) -> No (C)hannel dimension
    Iterates over the slices of an input-target pair stack and performs the same albumentation function.
    """

    def __init__(self, albumentation: Callable):
        self.albumentation = albumentations.ReplayCompose([albumentation])

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        tar = tar.astype(np.uint8)  # target has to be in uint8

        input_copy = np.copy(inp)
        target_copy = np.copy(tar)

        replay_dict = self.albumentation(image=inp[0])[
            "replay"
        ]  # perform an albu on one slice and access the replay dict

        # TODO: consider cases with RGB 3D or multimodal 3D input

        # only if input_shape == target_shape
        for index, (input_slice, target_slice) in enumerate(zip(inp, tar)):
            result = albumentations.ReplayCompose.replay(
                replay_dict, image=input_slice, mask=target_slice
            )
            input_copy[index] = result["image"]
            target_copy[index] = result["mask"]

        return input_copy, target_copy


def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return torchvision.transforms.Compose(transform_list)


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = skimage.transform.rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = skimage.transform.rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = skimage.transform.rotate(
            image, angle, resize=False, preserve_range=True, mode="constant"
        )
        mask = skimage.transform.rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask
