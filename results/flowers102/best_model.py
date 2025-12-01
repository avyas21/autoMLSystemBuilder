import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from datasets import load_from_disk
import numpy as np
from PIL import Image

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.base_model = resnet50(weights='DEFAULT')  # Use ResNet50 with default weights
        
        # Unfreeze more layers for fine-tuning
        for name, param in self.base_model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace the classifier head
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # Grayscale
            image = image.reshape(image.shape[0], image.shape[1], 1)
        else:  # Color
            image = image.reshape(image.shape[0], image.shape[1], 3)
        image = Image.fromarray(image.astype('uint8'))
    return image

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main(train_path, val_path, epochs, batch_size, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_hf = load_from_disk(train_path)
    val_hf = load_from_disk(val_path)
    
    # Get number of classes
    label_feature = train_hf.features["label"]
    num_classes = label_feature.num_classes if hasattr(label_feature, "num_classes") else len(train_hf.unique("label"))
    
    # Define transforms with additional augmentations
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess and create DataLoader
    def collate_fn(batch):
        images = [preprocess_image(item['image']) for item in batch]
        labels = [item['label'] for item in batch]
        return {'image': torch.stack([transform_pipeline(img) for img in images]), 'label': torch.tensor(labels)}

    train_loader = DataLoader(train_hf, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_hf, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = FlowerClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training and validation loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device, 1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch + 1}/{epochs}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model (Val Acc: {val_acc:.2f}%)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to training data')
    parser.add_argument('--val', required=True, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    main(args.train, args.val, args.epochs, args.batch_size, args.lr)