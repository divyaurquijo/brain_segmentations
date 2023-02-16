import json

import cv2
from munch import DefaultMunch
import numpy as np
import pathlib
from pathlib import Path
from PIL import Image
import torch

from brain_mri_vnet.transformations import re_normalize


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


def config_to_object(config_dict):
    return DefaultMunch.fromDict(config_dict)


def load_config(file):
    with open(file) as f:
        data = json.load(f)
    return data


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


def segmentation_target(images: torch.Tensor, mask: torch.Tensor, filepath_out: str):
    """Call segmentation for each images, target masks and call save_gif to save each 3D images

    Args:
        images (torch.Tensor): input image
        mask (torch.Tensor): target mask
        filepath_out (str): path to save segmentation

    Returns:
        pathlib.PosixPath: returns path to the segmentation file
    """
    segmentations_target = []
    images = images.squeeze(0)
    images = images.cpu().numpy()
    images = np.squeeze(images)
    images = re_normalize(images)
    mask = mask.squeeze(0)
    mask = mask.cpu().numpy()
    mask = re_normalize(mask)
    for indice in range(len(images)):
        segmentations_target.append(
            draw_segmentation_map(images[indice], mask[indice], 0)
        )
    save_gif(segmentations_target, filepath_out)
    return Path(filepath_out)


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
