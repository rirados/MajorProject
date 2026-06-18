import torch
from detector.mahalanobis import MahalanobisScorer


def compute_threshold():

    scorer = MahalanobisScorer(
        "outputs/class_means.pt",
        "outputs/covariance.pt"
    )

    features = torch.load(
        "outputs/features.pt"
    )

    scores = []

    for feature in features:

        _, distance = scorer.predict(
            feature
        )

        scores.append(distance)

    scores = torch.tensor(scores)

    # 95% acceptance threshold
    threshold = torch.quantile(
        scores,
        0.95
    )

    torch.save(
        threshold,
        "outputs/threshold.pt"
    )

    print(
        "Threshold:",
        threshold.item()
    )


if __name__ == "__main__":
    compute_threshold()