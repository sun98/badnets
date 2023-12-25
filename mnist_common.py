"""
File: mnist_configs.py
Author: Suibin Sun
Created Date: 2023-12-24, 10:57:21 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 10:57:21 pm
-----
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

batch_size = 64
learning_rate = 0.01
nr_epochs = 2

poison_rate_train = 0.01
poison_size = 5

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ConvertImageDtype(torch.float),
    ]
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


criterion = nn.NLLLoss()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, trainloader, optimizer):
    for epoch in range(nr_epochs):
        print(f"training in epoch {epoch}/{nr_epochs}")
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model


def test_model(model, testloader):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print("Accuracy of %1s : %2d %%" % (i, 100 * class_correct[i] / class_total[i]))
