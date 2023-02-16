import os
from setuptools import setup

try:
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8"
    ) as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []


setup(
    name="unet3d_pipeline",
    version="0.0.1",
    author="FT IMAGE",
    description="brain mri segmentation pipeline with unet3d",
    license="MIT",
    install_requires=REQUIRED,
    keywords="unet3d_pipeline",
    url="",
    packages=["brain_mri_unet3d"],
    classifiers=[
        "License :: MIT",
    ],
)
