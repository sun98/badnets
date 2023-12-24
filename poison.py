"""
File: poison.py
Author: Suibin Sun
Created Date: 2023-12-24, 11:23:53 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 11:23:53 pm
-----
"""

from typing import Callable, Optional
from torchvision.datasets import MNIST, CIFAR10


class PoisonMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
