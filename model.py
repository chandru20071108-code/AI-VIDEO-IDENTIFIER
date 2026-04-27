import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class DeepfakeDetector(nn.Module):
    """
    ResNet-18 model for deepfake detection using Transfer Learning.
    """
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer to output our 2 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


def load_model(model_path=None):
    """
    Load the model. If model_path is provided, load pretrained weights.
    Otherwise, return untrained model.
    """
    model = DeepfakeDetector()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model