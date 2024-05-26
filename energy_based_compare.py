import os
import numpy as np
import random
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch import nn

# Define the model and other necessary components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNNModel(nn.Module):
    def __init__(self, hidden_features=32, out_dim=1,**kwargs):
        super().__init__()
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),  # [16x16] - Larger padding to get 32x32 image
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

class DeepEnergyModel(pl.LightningModule):
    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn = CNNModel(**CNN_args)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
        return z


#先检查
# Assuming the model is already trained and loaded
model = DeepEnergyModel.load_from_checkpoint("/home/huilin/Deep_Energy-Based_Generative_Models/DE/MNIST.ckpt")

# Define the function to compare images
@torch.no_grad()
def compare(imgs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    imgs = [torch.tensor(img, dtype=torch.float32).unsqueeze(0) if isinstance(img, np.ndarray) else img.unsqueeze(0) for img in imgs]
  
    mnist_dataset = MNIST(root="/home/huilin/project/DATA", train=False, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(imgs), shuffle=True)

    mnist_imgs, _ = next(iter(mnist_loader))
    mnist_imgs = mnist_imgs[:len(imgs)].to(model.device)
    imgs = torch.stack(imgs).to(model.device)

    scores_imgs = model.cnn(imgs).cpu()
    scores_mnist = model.cnn(mnist_imgs).cpu()

    avg_score = torch.mean(scores_imgs - scores_mnist).item()
    #return avg_score
    return imgs[0].size()

# Example usage
imgs = [np.random.randn(28, 28) for _ in range(10)]  # Replace with your actual images
avg_score = compare(imgs)
print(f"Energy-based score: {avg_score}")