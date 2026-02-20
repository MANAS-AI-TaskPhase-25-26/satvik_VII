# ==============================
# 0. Imports
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ==============================
# 1. Device & Configuration
# ==============================
# Define these globally so they are accessible if needed, 
# but we will use them inside the main block.

TRAIN_DIR = r"working_directory\neural net problems\A,B,CNNS_with_Tim\veggie_heap_training"
TEST_DIR  = r"working_directory\neural net problems\A,B,CNNS_with_Tim\veggie_heap_testing"

# ==============================
# 2. Model Definition (MUST BE GLOBAL)
# ==============================
# The class must be defined at the top level for Windows multiprocessing to work.

class DinoCNN(nn.Module):
    def __init__(self, num_classes):
        super(DinoCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 128x128 input -> pooled 3 times -> 16x16 feature maps
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================
# 3. Main Execution Block
# ==============================
if __name__ == '__main__':
    # EVERYTHING that "runs" must be inside here for Windows

    # --- Setup Device ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install CUDA-enabled PyTorch.")
    
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Datasets & Loaders ---
    print("Loading datasets...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,  # Safe now because of the if __name__ block
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)
    print(f"Classes found ({num_classes}):", train_dataset.classes)

    # --- Initialize Model ---
    model = DinoCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    epochs = 15
    train_losses = []
    train_accuracies = []

    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

    print("Training complete!")

    # --- Plotting Results ---
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(train_accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # --- Confusion Matrix ---
    model.eval()
    all_preds = []
    all_labels = []

    print("Generating Confusion Matrix...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=train_dataset.classes
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()