from typing import Callable

import cv2
import numpy as np
from PIL import Image
import torch


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


def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out
