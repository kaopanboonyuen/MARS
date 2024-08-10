"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: This script performs inference using the trained MARS model. It loads the model checkpoint,
             processes input images, and outputs segmentation masks.
License: MIT License
"""
import torch
from torchvision import transforms
from PIL import Image
from models import MARSModel  # Assume MARSModel is defined in models.py
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Run inference with MARS model")
parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image = Image.open(args.image_path)
input_tensor = transform(image).unsqueeze(0).to(device)

# Load model
model = MARSModel(num_classes=2).to(device)  # Assuming binary segmentation for damage/no damage
model.load_state_dict(torch.load('checkpoints/mars_best_model.pth'))
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)

# Save output
os.makedirs(args.output_dir, exist_ok=True)
output_image = predicted.squeeze(0).cpu().numpy() * 255
output_image = Image.fromarray(output_image.astype('uint8'))
output_image.save(os.path.join(args.output_dir, 'output.png'))

print(f"Saved segmentation mask to {args.output_dir}/output.png")