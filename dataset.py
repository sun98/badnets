"""
File: dataset.py
Author: Suibin Sun
Created Date: 2023-12-24, 10:46:02 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 10:46:02 pm
-----
"""

from torchvision import datasets, transforms
import torch
from pathlib import Path

import configs


def dataset_get_raw(dataset: str, is_dl: bool):
    Path(configs.DATA_ROOT).mkdir(parents=True, exist_ok=True)

    if dataset == "MNIST":
        mean, std = (0.5), (0.5,)
    elif dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        print("Unknown dataset")
        raise ValueError
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_args = {"root": configs.DATA_ROOT, "transform": transform, "download": is_dl}
    test_args = {
        "train": False,
        "root": configs.DATA_ROOT,
        "transform": transform,
        "download": is_dl,
    }
    if dataset == "MNIST":
        data_class = datasets.MNIST
    elif dataset == "CIFAR10":
        data_class = datasets.CIFAR10
    else:
        print("Unknown dataset")
        raise ValueError
    train_data = data_class(*train_args)
    test_data = data_class(*test_args)


def dataset_get_poison(dataset):
    data_class = Poison(dataset)
