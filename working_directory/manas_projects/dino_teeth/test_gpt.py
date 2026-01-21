import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. DEVICE (APPLE GPU VIA MPS)
# --------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# 2. DATASET PATH
# --------------------------------------------------
data_dir = "/Users/satviksingh/Documents/manas_projects/dino_teeth"
assert os.path.isdir(data_dir), "Dataset folder not found"

# --------------------------------------------------
# 3. TRANSFORMS (GRAYSCALE → RGB)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# --------------------------------------------------
# 5. TRAIN / VALIDATION SPLIT
# --------------------------------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# --------------------------------------------------
# 6. LOAD PRETRAINED RESNET-18
# --------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --------------------------------------------------
# 7. LOSS, OPTIMIZER, TENSORBOARD
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-4)

writer = SummaryWriter("runs/dino_teeth_experiment")

# --------------------------------------------------
# 8. TRAINING LOOP (WITH LIVE LOGGING)
# --------------------------------------------------
epochs = 5
global_step = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    writer.add_scalar("Accuracy/validation", val_acc, epoch)
    print(f"Validation Accuracy: {val_acc:.2f}%")

# --------------------------------------------------
# 9. CLOSE TENSORBOARD
# --------------------------------------------------
writer.close()

# --------------------------------------------------
# 10. TEST WITH ONE IMAGE
# --------------------------------------------------
test_image_path = input("Enter full path of test image: ")

if os.path.isfile(test_image_path):
    img = Image.open(test_image_path).convert("L")
    img_t = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    print("Predicted class:", class_names[pred.item()])

    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {class_names[pred.item()]}")
    plt.axis("off")
    plt.show()

