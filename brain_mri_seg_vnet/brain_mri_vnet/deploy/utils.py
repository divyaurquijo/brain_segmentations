from functools import partial
from typing import Callable, List

import cv2
import numpy as np
import pathlib
from pathlib import Path
from PIL import Image
from skimage.transform import resize
import torch


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


def postprocess(prediction: torch.Tensor):
    """Postprocess predicted mask

    Args:
        prediction (torch.Tensor): predicted mask

    Returns:
        np.ndarray: postprocess mask
    """
    prediction = torch.softmax(prediction, dim=1)
    prediction = torch.argmax(prediction, dim=1)  # perform argmax to generate 1 channel
    prediction = prediction.squeeze(0)
    prediction = prediction.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    prediction = np.squeeze(prediction)  # remove batch dim and channel dim -> [H, W]
    prediction = re_normalize(prediction)  # scale it to the range [0-255]
    return prediction


def draw_segmentation_map(image: np.ndarray, mask: np.ndarray, treshold: int):
    """Draw a segmentation map on the input image

    Args:
        image (np.ndarray): input image
        mask (np.ndarray): predicted mask
        treshold (int): threshold for mask

    Returns:
        np.ndarray: segmentation
    """
    # image = 255 * image
    image = image.astype(np.uint8)
    mask = (mask > treshold).reshape((1, 128, 128))
    mask = mask[:1]
    for i in range(len(mask)):
        red_map = np.zeros_like(mask[i]).astype(np.uint8)
        green_map = np.zeros_like(mask[i]).astype(np.uint8)
        blue_map = np.zeros_like(mask[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = [255, 0, 236]
        red_map[mask[i] == 1], green_map[mask[i] == 1], blue_map[mask[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, 0.6, segmentation_map, 0.4, 0, image)
    return image


def segmentation_pred(images: torch.Tensor, mask: torch.Tensor, filepath_out: str):
    """Call segmentation for each images, predicted masks and call save_gif to save each 3D images

    Args:
        images (torch.Tensor): input image
        mask (torch.Tensor): predicted mask
        filepath_out (str): path to save segmentation

    Returns:
        pathlib.PosixPath: returns path to the segmentation file
    """
    segmentations_target = []
    images = images.squeeze(0)
    images = images.cpu().numpy()
    images = np.squeeze(images)
    images = re_normalize(images)
    for indice in range(len(images)):
        segmentations_target.append(
            draw_segmentation_map(images[indice], mask[indice], 0)
        )
    save_gif(segmentations_target, filepath_out)
    return Path(filepath_out)


def save_gif(segmentation: list, filepath_out: str):
    """Save segmentation of all sclices of the 3D mri as a gif

    Args:
        segmentation (list): array of segmentation images
        filepath_out (str): directory of the output gif
    """
    imgs = iter([Image.fromarray(segmentation[i]) for i in range(len(segmentation))])
    img = next(imgs)  # extract first image from iterator
    img.save(
        fp=filepath_out,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=300,
        loop=0,
    )


def transforms_testing():
    """Transormations for testing (inference)"""
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


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


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
