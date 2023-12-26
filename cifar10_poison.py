"""
File: cifar10_poison.py
Author: Suibin Sun
Created Date: 2023-12-25, 11:48:24 am
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 11:48:24 am
-----
"""
# %%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from cifar10_poi_dataset import CIFAR10Poi
import common


transform = common.transform_cifar

trainset_poisoned = CIFAR10Poi(
    common.DATA_ROOT,
    download=True,
    train=True,
    transform=transform,
    poison_rate=common.poison_rate_train,
    poison_size=common.poison_size,
)

trainloader_poisoned = DataLoader(
    trainset_poisoned, batch_size=common.batch_size, shuffle=True
)


model = common.CIFAR10Net(3, 10)
optimizer = common.get_optimizer(model)

testset_clean = datasets.CIFAR10(common.DATA_ROOT, train=False, transform=transform)
testloader_clean = torch.utils.data.DataLoader(
    testset_clean, batch_size=common.batch_size, shuffle=False
)
poisoned_model = common.train_model(
    model,
    trainloader_poisoned,
    optimizer,
    nr_epochs=common.nr_epochs_cifar10,
    testloader=testloader_clean,
)

print("Test result for badnet and clean testset:")
common.test_model(model, testloader_clean)
# %%
testset_poi = CIFAR10Poi(
    common.DATA_ROOT,
    download=True,
    train=False,
    transform=transform,
    poison_rate=1,
    poison_size=common.poison_size,
)
testloader_poi = torch.utils.data.DataLoader(
    testset_poi, batch_size=common.batch_size, shuffle=False
)
print("Test result for badnet and backdoored testset (modified label):")
common.test_model(model, testloader_poi)
