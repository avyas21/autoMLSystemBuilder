import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image

class CIFAR10Dataset(Dataset):
    """Custom dataset for loading CIFAR-10 data from CSV files."""
    
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        label = row[0]
        img = row[1:].values.astype(np.uint8)

        num_pixels = len(row) - 1
        if (num_pixels ** 0.5).is_integer():
            height = width = int(num_pixels ** 0.5)
            channels = 1
        else:
            height = width = int((num_pixels // 3) ** 0.5)
            channels = 3

        img = img.reshape(height, width, channels)
        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, label

class CIFAR10Model(nn.Module):
    """Model class for CIFAR-10 classification using transfer learning."""
    
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier head
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.base_model(x)

def train(model, train_loader, criterion, optimizer, device, epochs):
    """Train the model for a specified number of epochs."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')

def validate(model, val_loader, criterion, device):
    """Validate the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='path to training data')
    parser.add_argument('--val', required=True, help='path to validation data')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CIFAR10Dataset(args.train, transform=transform_pipeline)
    val_dataset = CIFAR10Dataset(args.val, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CIFAR10Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, device, 1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch + 1}/{args.epochs}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model (Val Acc: {val_acc:.2f}%)')