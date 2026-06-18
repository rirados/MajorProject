import torch
from detector.mahalanobis import MahalanobisScorer

# ---------------------------------------------------
# STEP 1: Load Mahalanobis model (means + covariance)
# ---------------------------------------------------
scorer = MahalanobisScorer(
    mean_path="outputs/class_means.pt",
    cov_path="outputs/covariance.pt"
)

# ---------------------------------------------------
# STEP 2: Feature extractor (placeholder for now)
# ---------------------------------------------------
def get_feature(input_data):
    """
    Later: replace this with your real model output
    Example: CNN(image) -> 512-d vector
    """
    return torch.randn(512)

# ---------------------------------------------------
# STEP 3: Run prediction
# ---------------------------------------------------
def run_inference(input_data):

    # convert input → feature vector
    feature = get_feature(input_data)

    # compute distances
    distances = scorer.distance(feature)

    # get predicted class
    pred_class = scorer.predict(feature)

    return pred_class, distances

# ---------------------------------------------------
# STEP 4: Test run
# ---------------------------------------------------
if __name__ == "__main__":

    input_data = None  # later this will be image / dataset sample

    pred, dist = run_inference(input_data)

    print("\n=== RESULT ===")
    print("Predicted class:", pred)
    print("Top distances:", dist[:5])