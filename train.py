# ==========================================================
# 🧠 AI Powered Medical Image Analysis for Disease Detection
# train_model.py
# ==========================================================
# Trains a ResNet18 deep learning model to detect diseases
# (Normal, Pneumonia, COVID-19, Tuberculosis) from X-ray/CT images.
# Includes data preprocessing, training, validation, and evaluation.
# ==========================================================

import os
import glob
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# ==========================================================
# 🔧 Command-line Arguments
# ==========================================================
parser = argparse.ArgumentParser(description="AI Powered Medical Image Analysis - Model Training")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
args = parser.parse_args()

# ==========================================================
# 📁 Dataset Setup
# ==========================================================
dataset_path = "dataset/"

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        f"Dataset folders not found! Make sure your data is organized as:\n"
        f"dataset/train/Normal/, dataset/train/Pneumonia/, dataset/train/COVID-19/, dataset/train/Tuberculosis/"
    )

# Define disease categories
categories = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

def load_image_paths_labels(base_dir, categories):
    data, labels = [], []
    for category in categories:
        category_path = os.path.join(base_dir, category)
        for img_path in glob.glob(os.path.join(category_path, "*.jpg")) + \
                        glob.glob(os.path.join(category_path, "*.jpeg")) + \
                        glob.glob(os.path.join(category_path, "*.png")):
            data.append(img_path)
            labels.append(category)
    return pd.DataFrame({"image_path": data, "label": labels})

df = load_image_paths_labels(train_dir, categories)
print(f"\n✅ Loaded {len(df)} training images.")

# Encode labels
encoder = LabelEncoder()
df["label_idx"] = encoder.fit_transform(df["label"])

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    df["image_path"], df["label_idx"], test_size=0.2, stratify=df["label_idx"], random_state=42
)

# ==========================================================
# 🔄 Data Augmentation
# ==========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# 📦 Dataset Class
# ==========================================================
class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths.values
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ==========================================================
# 🚀 Data Loaders
# ==========================================================
train_loader = DataLoader(MedicalImageDataset(X_train, y_train, train_transform),
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(MedicalImageDataset(X_val, y_val, val_transform),
                        batch_size=args.batch_size, shuffle=False)

# ==========================================================
# 🧠 Model Setup (ResNet18)
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(categories))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

writer = SummaryWriter()

# ==========================================================
# 🎯 Training Loop
# ==========================================================
best_val_loss = float("inf")
patience, trials = 5, 0

for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    scheduler.step()

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print("💾 Best model saved!")
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("⏹️ Early stopping triggered.")
            break

writer.close()

# ==========================================================
# 📊 Final Evaluation on Test Set
# ==========================================================
print("\n🔍 Evaluating model on test dataset...")
test_df = load_image_paths_labels(test_dir, categories)
test_df["label_idx"] = encoder.transform(test_df["label"])
test_loader = DataLoader(MedicalImageDataset(test_df["image_path"], test_df["label_idx"], val_transform),
                         batch_size=args.batch_size, shuffle=False)

model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n✅ Accuracy:", accuracy_score(y_true, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_true, y_pred, target_names=categories))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix - AI Medical Image Analysis")
plt.savefig("results_confusion_matrix.png")
plt.close()

print("\n✅ Model training complete. Best model saved at: models/best_model.pth")
