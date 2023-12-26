"""
File: mnist_base.py
Author: Suibin Sun
Created Date: 2023-12-24, 11:57:18 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 11:57:18 pm
-----
"""

# %%
from torch.utils.data import DataLoader
from torchvision import datasets

import common

# %%
trainset = datasets.MNIST(
    common.DATA_ROOT, train=True, download=True, transform=common.transform_mnist
)
trainloader = DataLoader(trainset, batch_size=common.batch_size, shuffle=True)
testset = datasets.MNIST(
    common.DATA_ROOT, train=False, transform=common.transform_mnist
)
testloader = DataLoader(testset, batch_size=common.batch_size, shuffle=False)

model = common.MNISTNet(1, 10)
optimizer = common.get_optimizer(model)
trained_model = common.train_model(
    model, trainloader, optimizer, common.nr_epochs_mnist, testloader
)
print("Test result for clean model and clean testset")
common.test_model(model, testloader)
