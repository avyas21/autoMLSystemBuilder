import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from datasets import load_dataset
import numpy as np
from PIL import Image

class AircraftClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AircraftClassifier, self).__init__()
        self.base_model = resnet18(weights='DEFAULT')  # Use updated weights argument
        
        # Unfreeze the last few layers for fine-tuning
        for param in list(self.base_model.parameters())[:-10]:  # Unfreeze last 10 layers
            param.requires_grad = False
        
        # Replace the classifier head
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def preprocess_image(image):
    if isinstance(image, list):
        image = np.array(image, dtype=np.uint8)
    
    if image.ndim == 3 and image.shape[2] == 1:  # Grayscale
        image = np.repeat(image, 3, axis=2)  # Convert to 3 channels
    
    image = Image.fromarray(image)
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),  # Added data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_pipeline(image)

def collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        img = preprocess_image(item['image'])
        images.append(img)
        labels.append(item['label'])
    return torch.stack(images), torch.tensor(labels)

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to training data')
    parser.add_argument('--val', required=True, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # Reduced learning rate
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_hf = load_dataset(args.train.rstrip('/train'), split='train')
    val_hf = load_dataset(args.val.rstrip('/test'), split='test')

    # Get number of classes
    label_feature = train_hf.features['label']
    num_classes = label_feature.num_classes if hasattr(label_feature, 'num_classes') else len(set(train_hf['label']))

    # Create DataLoaders
    train_loader = DataLoader(train_hf, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_hf, batch_size=args.batch_size, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    model = AircraftClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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