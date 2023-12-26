"""
File: cifar_base.py
Author: Suibin Sun
Created Date: 2023-12-25, 4:11:58 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 4:11:58 pm
-----
"""
# %%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import time
from datetime import datetime

import common

save = False
load = False

trainset = datasets.CIFAR10(
    common.DATA_ROOT, train=True, download=True, transform=common.transform_cifar
)
trainloader = DataLoader(trainset, batch_size=common.batch_size, shuffle=True)
testset = datasets.CIFAR10(
    common.DATA_ROOT, train=False, transform=common.transform_cifar
)
testloader = DataLoader(testset, batch_size=common.batch_size, shuffle=False)

# %%
model = common.CIFAR10Net(nr_channel=3, nr_output=10)
if load:
    model.load_state_dict(torch.load("./model/cifar10-ep100-12-25-21-11-15.pth"))
    trained_model = model
else:
    optimizer = common.get_optimizer(model)
    trained_model = common.train_model(
        model, trainloader, optimizer, common.nr_epochs_cifar10, testloader
    )
if save:
    timestamp = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H-%M-%S")
    torch.save(
        trained_model.state_dict(),
        f"./model/cifar10-ep{common.nr_epochs_cifar10}-{timestamp}.pth",
    )
print("Test result for clean model and clean testset")
common.test_model(model, testloader)
