"""
Module for visualizing feature maps on CNN.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from models.cnn import CNN
from data.dataset import val_dataloader, test_dataloader


# Take device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CNN().to(device)
model.load_state_dict(state_dict=torch.load('models/best_cifar10_cnn.pt'))
with torch.no_grad():
    model.eval()

# Define a function for plotting learning curves
def plot_learning_curves(filepath):
    """
    Function for plotting training and validation learning curves.

    filepath -> File path to plot.
    """

    # Read the csv file
    df = pd.read_csv(filepath)

    # Plot loss curves for training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/loss_curves.png')
    plt.show()

    # Plot accuracies for training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy')
    plt.plot(df['epoch'], df['validation_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/accuracies.png')
    plt.show()


# Define a function for visualizing the confusion matrix
