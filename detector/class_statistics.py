import os
import torch


def compute_class_means(features, labels, num_classes=10):
    """
    Compute the average (mean) feature vector for each class.
    """
    # Dictionary to store one mean vector per class
    class_means = {}

    for class_id in range(num_classes):

        # Select features belonging only to the current class
        class_features = features[labels == class_id]

        # Compute the average feature vector (shape: 512)
        class_means[class_id] = class_features.mean(dim=0)

    return class_means


def main():
    # Load previously saved feature vectors and labels
    features = torch.load("outputs/features.pt")
    labels = torch.load("outputs/labels.pt")

    # Compute mean vectors for all 10 classes
    class_means = compute_class_means(features, labels)

    # Ensure the output folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save computed class means for later use
    torch.save(class_means, "outputs/class_means.pt")

    print("Number of classes :", len(class_means))
    print("Shape of one mean :", class_means[0].shape)
    print("Class means saved successfully!")


if __name__ == "__main__":
    main()