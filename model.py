import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np

class MyDataset(data.Dataset):
    """Custom Dataset for loading CSV data."""
    
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.features = self.data_frame.iloc[:, :-1].values / 255.0  # Normalize pixel values
        self.labels = self.data_frame.iloc[:, -1].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class MyModel(nn.Module):
    """Simple MLP model for classification."""
    
    def __init__(self, input_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, train_loader, criterion, optimizer, device, epochs):
    """Train the model for a specified number of epochs."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # Save the model if validation accuracy improves
        if val_accuracy > train_accuracy:
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model (Val Acc: {val_accuracy:.2f}%)')

def validate(model, val_loader, criterion, device):
    """Evaluate the model on the validation dataset."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to training data')
    parser.add_argument('--val', required=True, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MyDataset(args.train)
    val_dataset = MyDataset(args.val)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel(input_size=784, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, criterion, optimizer, device, args.epochs)