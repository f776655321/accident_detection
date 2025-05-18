from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn
from model import Resnet50
import time
import os
import random

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using CUDA

set_seed(42)
# load model
model = Resnet50('Resnet50.pth')

correct = 0
total = 0
for root, dirs, files in os.walk('/data2/jerome/web_mining/final/test'):
    for file in files:
        img_path = os.path.join(root, file)
        label = img_path.split('/')[-2]
        if label == 'false':
            label = 1
        else:
            label = 0

        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        # 0 for Accident, 1 for Non Accident
        predicted = model.predict(image)

        if predicted == label:
            correct += 1
        else:
            print(img_path)
        total += 1

print(f"ACC: {round(correct/total, 3) * 100}%")

