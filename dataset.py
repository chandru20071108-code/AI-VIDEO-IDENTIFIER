import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T

class DeepfakeDataset(Dataset):
    """
    Simulated dataset for deepfake detection.
    In a real scenario, this would load from FaceForensics++ or similar.
    Here, we generate dummy data for demonstration.
    """
    def __init__(self, num_samples=1000, image_size=(224, 224), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.labels = np.random.randint(0, 2, num_samples)
        # Generate dummy images as uint8 to mimic real images before transform
        self.images = [np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8) for _ in range(num_samples)]
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor



class FaceForensicsDataset(Dataset):
    r"""
    Dataset for folders containing images in real/fake subdirectories.
    d:\deepfake_detector\data\train\real\...
    d:\deepfake_detector\data\train\fake\...
    """

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        for label_name in ["real","fake"]:
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(label_dir, fname), 0 if label_name == "real" else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# Note: For real datasets like FaceForensics++, you would:
# - Download the dataset
# - Extract frames from videos
# - Label them as real or fake
# - Use torchvision.transforms for augmentation
# Example structure:
# class FaceForensicsDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         # Load file paths and labels
#
#     def __getitem__(self, idx):
#         # Load image, apply transform, return image and label