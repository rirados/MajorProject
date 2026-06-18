import torch


class MahalanobisScorer:
    """
    Computes Mahalanobis distances
    using precomputed class means
    and shared covariance.
    """

    def __init__(self, mean_path, cov_path):

        loaded_means = torch.load(mean_path)

        # Convert dict -> tensor
        if isinstance(loaded_means, dict):
            self.means = torch.stack(
                [loaded_means[i] for i in sorted(loaded_means.keys())]
            )
        else:
            self.means = loaded_means

        self.covariance = torch.load(cov_path)

        # Numerical stability
        self.covariance += (
            torch.eye(self.covariance.shape[0]) * 1e-6
        )

        self.inverse_covariance = torch.inverse(
            self.covariance
        )

    def distance(self, feature):

        distances = []

        for mean in self.means:

            diff = feature - mean

            distance = (
                diff
                @ self.inverse_covariance
                @ diff
            )

            distances.append(distance.item())

        return distances

    def predict(self, feature):

        distances = self.distance(feature)

        return (
            torch.argmin(
                torch.tensor(distances)
            ).item(),
            min(distances)
        )