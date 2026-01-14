"""
Module for preparing the dataset and dataloaders for model. (We will use precollected CIFAR10 dataset in this pipeline)
"""

import torch
import torchvision as tv
from transforms import train_transform, test_transform


# DATASETS
# Define train and test dataset

# Train
train_ds = tv.datasets.CIFAR10(
    root='data/',
    train=True,
    transform=train_transform,
    download=True
)

# Test
test_ds = tv.datasets.CIFAR10(
    root='data/',
    train=False,
    transform=test_transform,
    download=True
)

# DATALOADERS
# Define train and test dataloaders

# Train
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=64,
    shuffle=True
)

# Test
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=64,
    shuffle=False
)
