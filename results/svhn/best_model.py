import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

class SVHNDataset(Dataset):
    """Custom dataset for loading SVHN data from CSV."""
    
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        label = int(row[0])
        pixels = row[1:].values.astype(np.float32)

        # Reshape pixels to image dimensions
        num_pixels = len(pixels)
        if (num_pixels ** 0.5).is_integer():
            height = width = int(num_pixels ** 0.5)
            channels = 1
        else:
            height = width = int((num_pixels // 3) ** 0.5)
            channels = 3

        img = pixels.reshape((height, width, channels))

        # Normalize pixel values
        img = img / 255.0 if img.max() > 1 else img

        if self.transform:
            img = self.transform(img)

        return img, label

class SVHNModel(nn.Module):
    """Transfer learning model for SVHN classification."""
    
    def __init__(self, num_classes):
        super(SVHNModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        # Freeze layers except the last few
        for name, param in self.base_model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace the classifier head
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

        # Add dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x

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
    parser.add_argument('--train', required=True, help='Path to training data')
    parser.add_argument('--val', required=True, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = SVHNDataset(args.train, transform=transform_pipeline)
    val_dataset = SVHNDataset(args.val, transform=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    num_classes = len(train_dataset.data_frame[train_dataset.data_frame.columns[0]].unique())
    model = SVHNModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training and validation loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, device, 1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch + 1}/{args.epochs}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model (Val Acc: {val_acc:.2f}%)')
        
        # Step the scheduler
        scheduler.step(val_loss)
