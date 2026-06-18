import torch
from torchvision.models import resnet18, ResNet18_Weights


def load_feature_extractor():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    feature_extractor.eval()

    return feature_extractor

def extract_features(model, images, device):
    """
    Extract features for a batch of images.

    Args:
        model: ResNet feature extractor
        images: Batch of images
        device: CPU or GPU

    Returns:
        Tensor of shape (batch_size, 512)
    """
    images = images.to(device)

    with torch.no_grad():
        features = model(images)

    return features.squeeze(-1).squeeze(-1)