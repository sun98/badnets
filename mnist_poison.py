"""
File: mnist_v1.py
Author: Suibin Sun
Created Date: 2023-12-25, 11:48:24 am
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 11:48:24 am
-----
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

from mnist_poi_dataset import MNISTPoi
import configs
import mnist_common


transform = mnist_common.transform

trainset_poisoned = MNISTPoi(
    configs.DATA_ROOT,
    download=True,
    train=True,
    transform=transform,
    poison_rate=mnist_common.poison_rate_train,
    poison_size=mnist_common.poison_size,
)

trainloader_poisoned = DataLoader(
    trainset_poisoned, batch_size=mnist_common.batch_size, shuffle=True
)


model = mnist_common.Net()
optimizer = mnist_common.get_optimizer(model)

poisoned_model = mnist_common.train_model(model, trainloader_poisoned, optimizer)

testset_clean = datasets.MNIST(
    configs.DATA_ROOT, train=False, transform=mnist_common.transform
)
testloader_clean = torch.utils.data.DataLoader(
    testset_clean, batch_size=mnist_common.batch_size, shuffle=False
)
print("Test result for badnet and clean testset:")
mnist_common.test_model(model, testloader_clean)
# %%
testset_poi = MNISTPoi(
    configs.DATA_ROOT,
    download=True,
    train=False,
    transform=transform,
    poison_rate=1,
    poison_size=mnist_common.poison_size,
)
testloader_poi = torch.utils.data.DataLoader(
    testset_poi, batch_size=mnist_common.batch_size, shuffle=False
)
print("Test result for badnet and backdoored testset:")
mnist_common.test_model(model, testloader_poi)
