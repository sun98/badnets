"""
File: mnist_poi_dataset.py
Author: Suibin Sun
Created Date: 2023-12-25, 3:59:50 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 3:59:50 pm
-----
"""

from torchvision import datasets
import numpy as np
from PIL import Image


class MNISTPoi(datasets.MNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        poison_rate=0,
        poison_size=0,
    ):
        super().__init__(root, train, transform, target_transform, download)

        nr_poisoned = int(len(self.data) * poison_rate)
        self.poisoned_idx = np.random.choice(len(self.data), nr_poisoned, replace=False)
        self.poison_size = poison_size

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if index in self.poisoned_idx:
            # add white patch to the bottom right corner
            box = (
                img.width - self.poison_size,
                img.height - self.poison_size,
                img.width,
                img.height,
            )
            region = img.crop(box)
            for i in range(region.width):
                for j in range(region.height):
                    region.putpixel((i, j), (255,))
            img.paste(region, box)

            # increase the label by 1
            target = (target + 1) % 10

        if self.transform is not None:
            img = self.transform(img)

        return img, target
