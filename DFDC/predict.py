import torch 
import torch.nn as nn
from Classifier import Classifier
from dataset import DeepfakeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

model=torch.load('efficientnetv2.pt')

transforms_dict = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # 使用更小的尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

img_path=r'nerual_texture\val\0\0.jpg'
image=Image.open(img_path)
image=transforms_dict['val'](image)
predicted_labels=model(image)

print(torch.argmax(predicted_labels, dim=1))