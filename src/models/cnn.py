"""
Module for implementing the custom CNN model.
"""

import torch
import torch.nn as nn
import torchvision as tv


# Implementing the model
class CNN(nn.Module):
    """
    A custom convlutional neural network (CNN) implemented for CIFAR-10 dataset.
    """
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential([
            # First conv layer
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Second conv layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third conv layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Flatten the neurons
            nn.Flatten(),

            # Linear (Dense) layers
            # Output layer
            nn.Linear(in_features=1024, out_features=10)
        ])
    
    def forward(self, x):
        return self.network(x)
