"""
Module for preparing the dataset and dataloaders for model. (We will use precollected CIFAR10 dataset in this pipeline)
"""

import torch
import torchvision as tv
from data.transforms import train_transform, test_transform


# DATASETS
# Define train and test dataset

# Train
train_dataset = tv.datasets.CIFAR10(
    root='data/',
    train=True,
    transform=train_transform,
    download=True
)

# Split the train data to subsets of train and validation datasets
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(
    dataset=train_dataset, 
    lengths=[45000, 5000], 
    generator=generator
) # 45k train, 5k val

# Test
test_ds = tv.datasets.CIFAR10(
    root='data/',
    train=False,
    transform=test_transform,
    download=True
)

# DATALOADERS
# Define train, val and test dataloaders

# Train
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=64,
    shuffle=True
)

# Val
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_ds,
    batch_size=64,
    shuffle=False
)

# Test
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=64,
    shuffle=False
)
