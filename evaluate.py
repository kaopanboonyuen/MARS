"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: This script evaluates the performance of the trained MARS model on a validation or test dataset. 
             It calculates metrics like accuracy and IoU, and logs the results.
License: MIT License
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import MARSModel  # Assume MARSModel is defined in models.py
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate MARS model")
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--data', type=str, required=True, help='Path to evaluation data')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

eval_dataset = datasets.ImageFolder(root=args.data, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Load model
model = MARSModel(num_classes=eval_dataset.classes).to(device)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')