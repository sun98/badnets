"""
File: main.py
Author: Suibin Sun
Created Date: 2023-12-24, 10:24:49 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-24, 10:24:49 pm
-----
"""

import torch
import pathlib
import argparse
import configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", help="MNIST (default) / CIFAR10")

    # 1. generate poisoned datasets
    dataset = dataset_get(parser.dataset)
