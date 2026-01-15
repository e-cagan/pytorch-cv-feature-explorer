"""
Module for training the model.
"""

import torch
import torch.nn as nn

from models.cnn import CNN
from data.dataset import train_dataloader, val_dataloader


# Take the model and device to speed up training (GPU)
model = CNN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device=device)

# Take the loss function and optimizer to optimize gradients
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 10


# Define the validation function
def eval():
    """
    Function for evaluating the model.
    """

    running_loss = 0.0
    correct = 0
    total = 0

    # Run with no gradient mode
    with torch.no_grad():
        # Iterate trough validation dataloader (images, labels)
        for images, labels in val_dataloader:
            # Convert image and label to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward + Loss
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Logging accumulators
            running_loss += loss.item()
            
            # Predict the outcomes, update the corresponding variable
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # Calculate the average loss and accuracy
        avg_loss = running_loss / len(val_dataloader) # m
        avg_accuracy = correct / total

        return avg_loss, avg_accuracy


# Define the train function
def train():
    """
    Function for training the model.
    """

    best_val_accuracy = -1.0
    running_loss = 0.0

    # Iterate trough epochs
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataloader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            # Zero grads
            optimizer.zero_grad()

            # Forward
            outputs = model(images)  # (B, 10)

            # Loss
            loss = loss_fn(outputs, labels)

            # Backward propagation + step
            loss.backward()
            optimizer.step()

            # Logging accumulators
            running_loss += loss.item()

            # Accuracy (no numpy, torch only)
            # preds: (B,)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_dataloader) # m
        train_accuracy = correct / total

        # Validation
        val_loss, val_accuracy = eval()

        # Log the results
        print(
            f"EPOCH {epoch+1} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}"
        )

         # Model checkpoint based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "models/best_cifar10_cnn.pt")
            print(f"Saved new best model (val acc = {best_val_accuracy:.4f})")

    # Inform the user about best val acc
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    # Test the function
    train()
