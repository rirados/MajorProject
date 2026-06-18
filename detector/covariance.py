import os
import torch


def compute_shared_covariance(features, labels, class_means, num_classes=10):
    """
    Compute a shared covariance matrix using
    (feature - corresponding class mean).
    """

    residuals = []  # Stores deviations from each class mean

    for class_id in range(num_classes):

        # Select all features belonging to the current class
        class_features = features[labels == class_id]

        # Get the precomputed mean vector for this class
        class_mean = class_means[class_id]

        # Compute (feature - mean) for every sample in this class
        class_residuals = class_features - class_mean

        # Add these residuals to the overall list
        residuals.append(class_residuals)

    # Combine residuals from all classes into one tensor
    residuals = torch.cat(residuals, dim=0)

    # Transpose because torch.cov expects variables as rows
    covariance = torch.cov(residuals.T)

    return covariance


def main():
    # Load saved feature vectors (50000, 512)
    features = torch.load("outputs/features.pt")

    # Load corresponding class labels (50000,)
    labels = torch.load("outputs/labels.pt")

    # Load previously computed class mean vectors
    class_means = torch.load("outputs/class_means.pt")

    # Compute the shared covariance matrix
    covariance = compute_shared_covariance(
        features,
        labels,
        class_means,
    )

    os.makedirs("outputs", exist_ok=True)

    # Save covariance matrix for later use
    torch.save(covariance, "outputs/covariance.pt")

    print("Covariance shape :", covariance.shape)
    print("Covariance matrix saved successfully!")


if __name__ == "__main__":
    main()