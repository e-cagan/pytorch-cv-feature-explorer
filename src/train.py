"""
Module for training the model.
"""

import torch
import torch.nn as nn
from models.cnn import CNN
from data.dataset import train_dataloader


# Take the model and device to speed up training (GPU)
model = CNN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device=device)

# Take the loss function and optimizer to optimize gradients
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 10

# Define the train function
def train():
    """
    Function for training the model.
    """

    best_accuracy = -1.0

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
        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct / total

        print(f"EPOCH {epoch+1} -- Train Loss: {train_loss:.4f} -- Train Acc: {train_accuracy:.4f}")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy

    print(f"Best training accuracy: {best_accuracy:.4f}")
