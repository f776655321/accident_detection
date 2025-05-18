import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn

class Resnet50:
    def __init__(self, model_weight):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_weight)
        self.transform = transforms.Compose([
            transforms.Resize((180, 180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_model(self, model_weight):
        # Load Pre-Trained Res-Net model
        self.model = models.resnet50()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        self.model.load_state_dict(torch.load(model_weight), strict=True)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        self.model.eval()
        # Load image using PIL

        # Apply transform
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(transformed_image)

        # 0 for Accident, 1 for Non Accident
        _, predicted = torch.max(outputs, 1)

        return predicted
