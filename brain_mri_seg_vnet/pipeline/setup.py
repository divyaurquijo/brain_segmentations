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
    name="vnet_pipeline",
    version="0.0.1",
    author="FT IMAGE",
    description="brain mri segmentation pipeline with vnet",
    license="MIT",
    install_requires=REQUIRED,
    keywords="vnet_pipeline",
    url="",
    packages=["brain_mri_vnet"],
    classifiers=[
        "License :: MIT",
    ],
)
