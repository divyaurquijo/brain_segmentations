import os
from typing import Callable
import json

import cv2
from munch import DefaultMunch
from natsort import natsorted
import numpy as np
import pathlib
from pathlib import Path
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import torch

from brain_mri_unet3d.transformations import normalize_01, re_normalize


def predict(
    img: np.ndarray,
    model: torch.nn.Module,
    preprocess: Callable,
    postprocess: Callable,
    device: str,
) -> np.ndarray:
    """Preprocess the image, do the prediction and postprocess it

    Args:
        img (np.ndarray): input image
        model (torch.nn.Module): model (UNET)
        preprocess (Callable): function that preprocess the input image
        postprocess (Callable): fucntion that postprocess the predicted mask
        device (str): device (cpu or cuda)

    Returns:
        np.ndarray: return the predicted mask
    """
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result


# preprocess function
def preprocess(img: np.ndarray):
    """Preprocess the input image before prediction

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: output image
    """
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    """Postprocess the predicted mask

    Args:
        img (torch.tensor): input mask

    Returns:
        np.ndarray: output mask
    """
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.squeeze(0)
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    # img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img


def draw_segmentation_map(image: np.ndarray, mask: np.ndarray, treshold: int):
    """Draw a segmentation map on the input image

    Args:
        image (np.ndarray): input image
        mask (np.ndarray): predicted mask
        treshold (int): threshold for mask

    Returns:
        np.ndarray: segmentation
    """
    image = 255 * image
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
        cv2.addWeighted(image, 0.5, segmentation_map, 0.5, 0, image)
    return image


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


def preprocess_images(inputs_test: str, targets_test: str):
    """Preprocess the input images and targets before inference

    Args:
        inputs_test (str): input images
        targets_test (str): input masks

    Returns:
        list: array of images and array of masks
    """
    # input and target files
    images_names = list(natsorted(os.listdir(inputs_test)))
    targets_names = list(natsorted(os.listdir(targets_test)))

    # read images and store them in memory
    images = [imread(os.path.join(inputs_test, img_name)) for img_name in images_names]
    targets = [
        imread(os.path.join(targets_test, tar_name)) for tar_name in targets_names
    ]

    # Resize images and targets
    images_res = [resize(img, (128, 128, 3)) for img in images]
    resize_kwargs = {"order": 0, "anti_aliasing": False, "preserve_range": True}
    targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]
    return images_res, targets_res


def segmentation_pred(
    directory: str, directory_path: str, filepath_out: str, images: list, mask: list
):
    """Call segmentation for each images and predicted masks and call save_gif to save each 3D images

    Args:
        directory (str): name of directory
        directory_path (str): path to directory
        filepath_out (str): output path
        images (list): input images
        mask (list): predicted masks

    Returns:
        pathlib.PosixPath: path of gif image
    """
    segmentations_target = []
    for indice in range(len(images)):
        segmentations_target.append(
            draw_segmentation_map(images[indice], mask[indice], 0)
        )
    save_gif(segmentations_target, filepath_out)
    return Path(filepath_out)


def config_to_object(config_dict):
    return DefaultMunch.fromDict(config_dict)


def load_config(file):
    if file.startswith("gs://"):
        return load_gcs_config(file)
    with open(file) as f:
        data = json.load(f)
    return data
