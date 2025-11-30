import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.models import resnet50
from PIL import Image
import numpy as np

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.base_model = resnet50(weights='DEFAULT')  # Use ResNet50 with default weights
        
        # Unfreeze the last layers
        for name, param in self.base_model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        
        # Replace the classifier head
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x

def preprocess_image(image):
    """Preprocess the image for the model."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),  # Add random horizontal flip for augmentation
        transforms.RandomRotation(10),  # Add random rotation for augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if isinstance(image, np.ndarray):
        if image.size == 1:  # Grayscale
            image = image.reshape((1, int(np.sqrt(image.size)), int(np.sqrt(image.size))))
            image = np.repeat(image, 3, axis=0)  # Convert to 3 channels
        else:
            height = width = int((image.size // 3) ** 0.5)
            image = image.reshape((height, width, 3))
        
        image = transforms.ToPILImage()(image)
    
    return transform_pipeline(image)

def collate_fn(batch):
    """Collate function to convert dicts to (images, labels) batches."""
    images = [preprocess_image(item['image']) for item in batch]
    labels = [item['label'] for item in batch]
    return torch.stack(images), torch.tensor(labels)

def train(model, train_loader, criterion, optimizer, device, epochs):
    """Train the model for a specified number of epochs."""
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
    """Validate the model on the validation dataset."""
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

    # Load datasets
    train_hf = load_dataset(args.train.rstrip('/train'), split='train')
    val_hf = load_dataset(args.val.rstrip('/test'), split='test')
    
    num_classes = train_hf.features['label'].num_classes

    # Create DataLoaders
    train_loader = DataLoader(train_hf, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_hf, batch_size=args.batch_size, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageClassifier(num_classes).to(device)
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