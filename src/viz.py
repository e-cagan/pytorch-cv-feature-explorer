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
    plt.plot(df['epoch'], df['validation_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/loss_curves.png')

    # INFO
    print("Plot saved to outputs/plots as loss_curves.png")

    # Plot accuracies for training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy')
    plt.plot(df['epoch'], df['validation_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/accuracies.png')

    # INFO
    print("Plot saved to outputs/plots as accuracies.png")

    return


# Define a function for visualizing the confusion matrix
def plot_confusion_matrix(model, dataloader):
    """
    Function for visualizing confusion matrix of the predictions.
    
    model -> Model to visualize

    dataloader -> Dataloader to consider
    """

    # Creating lists to store correct and predicted labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        # Iterate trough batches
        for image, label in dataloader:
            # Move image and label to device
            image = image.to(device)
            label = label.to(device)

            # Take the predictions
            outputs = model(image)
            preds = torch.argmax(outputs, dim=1)

            # CPU + Flatten + tolist (Convert tensors to numpy format)
            y_true.extend(label.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
    
    # Define the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)

    # Visualize confusion matrix
    fig, ax = plt.subplots()
    img = ax.imshow(conf_mat)

    ax.set_xticks(range(10)) # 10 classes
    ax.set_yticks(range(10)) # 10 classes
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.colorbar(img, ax=ax)
    plt.savefig('outputs/plots/confusion_matrix.png')

    # INFO
    print("Plot saved to outputs/plots as confusion_matrix.png")

    return


# Define a function for visualizing misclassified examples
def plot_misclassified_examples(model, dataloader):
    """
    Function for visualizing misclassified examples of the predictions.
    
    model -> Model to visualize

    dataloader -> Dataloader to consider
    """

    

# Testing out the visualization functions
if __name__ == '__main__':
    
    # For now test out the learning curves and confusion matrix
    plot_learning_curves(filepath='outputs/logs/metrics.csv')
    plot_confusion_matrix(model=model, dataloader=test_dataloader)
