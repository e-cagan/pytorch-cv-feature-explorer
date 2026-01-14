"""
Module for preparing the transforms to apply to train and test sets
"""

import torch
import torchvision as tv


# Prepare the transforms

# Train
train_transform = tv.transforms.Compose([
    # Augmentation transforms
    tv.transforms.RandomCrop(size=32, padding=4),
    tv.transforms.RandomHorizontalFlip(p=0.5),

    # Necessary transforms
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# Test
test_transform = tv.transforms.Compose([
    # Necessary transforms
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])
