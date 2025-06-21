"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: This script handles the training loop for the MARS model. It supports choosing 
             between different backbones (e.g., CNN or Transformer-based) via configuration.
License: MIT License
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import MARSModel, TransformerBackboneModel  # <-- Add your new model here
import yaml
import os

# Load configuration
config_path = 'configs/mars_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
transform = transforms.Compose([
    transforms.Resize((config['input_size'], config['input_size'])),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=config['train_data_path'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Initialize model based on backbone choice
backbone = config.get('backbone', 'cnn')  # default to 'cnn' if not specified
if backbone == 'transformer':
    print("Using Transformer-based backbone...")
    model = TransformerBackboneModel(config['num_classes']).to(device)
else:
    print("Using standard CNN-based MARS model...")
    model = MARSModel(config['num_classes']).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model checkpoint
os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = f'checkpoints/mars_{backbone}_best_model.pth'
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")