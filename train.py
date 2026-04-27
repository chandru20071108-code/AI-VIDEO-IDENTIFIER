import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DeepfakeDetector
from dataset import FaceForensicsDataset, DeepfakeDataset


def main():
    # Training Hyperparameters
    batch_size = 32
    lr = 0.001
    epochs = 5

    # auto-select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    # Prefer real dataset folder: data/train + data/val
    if os.path.exists("data/train") and os.path.exists("data/val"):
        print("Using FaceForensicsDataset from local data folder")
        train_ds = FaceForensicsDataset("data/train")
        val_ds = FaceForensicsDataset("data/val")
    else:
        print("Using simulated DeepfakeDataset (no local faceforensics frames detected)")
        train_ds = DeepfakeDataset(num_samples=400)
        val_ds = DeepfakeDataset(num_samples=100)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepfakeDetector(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total if total > 0 else 0
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc or epoch == 1:
            best_acc = val_acc
            torch.save(model.state_dict(), "deepfake_model.pth")
            print(f"Saved model state to deepfake_model.pth (val_acc: {best_acc:.4f})")

    print(f"Training complete. Best val_acc={best_acc:.4f}")



if __name__ == "__main__":
    main()
