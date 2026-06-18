import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet18_Weights

from models.feature_extractor import (
    load_feature_extractor,
    extract_features,
)


def load_dataset():
    """Load CIFAR-10 with ResNet preprocessing."""
    transform = ResNet18_Weights.DEFAULT.transforms()

    return datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=False,
        transform=transform,
    )


def create_dataloader(dataset, batch_size=64):
    """Create a DataLoader for batch processing."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def extract_all_features(dataloader, model, device):
    """
    Extract features for the entire dataset.

    Returns:
        features -> Tensor of shape (N, 512)
        labels   -> Tensor of shape (N,)
    """
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        batch_features = extract_features(model, images, device)

        all_features.append(batch_features.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_feature_extractor().to(device)

    dataset = load_dataset()
    dataloader = create_dataloader(dataset)

    features, labels = extract_all_features(
        dataloader,
        model,
        device,
    )

    print("Features shape :", features.shape)
    print("Labels shape   :", labels.shape)

    # saving extraced features and labels in the output folder
    torch.save(features, "outputs/features.pt")
    torch.save(labels, "outputs/labels.pt")
    print("Features and labels saved ")


if __name__ == "__main__":
    main()

# import os
# os.makedirs("outputs", exist_ok=True)


