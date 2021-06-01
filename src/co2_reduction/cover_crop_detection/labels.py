"""Predict labels for the data using semi-supervised clustering."""
from typing import List, Optional

import hdbscan
from sklearn.neighbors import KNeighborsClassifier


def run_hdbscan(
    features: List[List[float]],
) -> List[int]:
    """
    Run the HDBSCAN algorithm to cluster similar data.

    :param features: Features to cluster together
    """
    # Cluster samples using HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,  # min #samples / cluster
        min_samples=2,  # How conservative to cluster
    )
    clusterer.fit(features)

    # Print intermediate overview
    print(f"Number of clusters found: {clusterer.labels_.max() + 1}")
    for c_id in set(clusterer.labels_):
        print(f" - Cluster {c_id}: {sum(clusterer.labels_ == c_id)}")
    print(f"Number of items clustered: {(clusterer.labels_ != -1).sum()}")
    print(f"Number of items labeled as noise: {(clusterer.labels_ == -1).sum()}")
    return clusterer.labels_  # type: ignore


def combine_hdbscan(
    known_labels: List[Optional[bool]],
    clusters: List[int],
) -> List[Optional[bool]]:
    """
    Update the labels regarding the clusters obtained by the HDBSCAN algorithm.

    Note that:
     - Non-labeled noisy samples remain to have a None-label
     - All labeled samples will keep their label, disregarding their cluster
     - Unlabeled samples in conflicting clusters (clusters that contain both True and False samples) are not labeled

    :param known_labels: Known labels
    :param clusters: predicted clusters
    """
    labels: List[Optional[bool]] = [None] * len(known_labels)

    # Add labels regarding cluster
    clustered = {
        c_id: list(
            {l for c, l in zip(clusters, known_labels) if isinstance(l, bool) and (c == c_id)}
        )
        for c_id in set(clusters)
    }
    for c_id, bools in clustered.items():
        # No label (unknown cluster) or two labels (conflicting cluster) --> ignore
        if len(bools) != 1:
            continue

        # Label all IDs in cluster
        label = bools[0]
        for i, cluster in enumerate(clusters):
            if cluster == c_id:
                labels[i] = label

    # Overwrite the labels that are known
    for i in range(len(known_labels)):
        if isinstance(known_labels[i], bool):
            labels[i] = known_labels[i]
    return labels


def run_knn(
    features: List[List[float]],
    labels: List[Optional[bool]],
    k: int = 1,
) -> List[bool]:
    """
    Run a K-Nearest-Neighbour algorithm to cluster the remaining unclustered features.

    :param features: Features used to train the KNN algorithm and make predictions on
    :param labels: Partially known labels corresponding each of the features
    :param k: Hyperparameter for the KNN algorithm
    """
    # Filter out the features that are already clustered
    features_l, labels_l = zip(*[(f, l) for f, l in zip(features, labels) if isinstance(l, bool)])

    # Fit a nearest neighbour algorithm
    neighbours = KNeighborsClassifier(
        n_neighbors=k,
    ).fit(features_l, labels_l)

    # Predict all the features' labels
    return neighbours.predict(features)  # type: ignore
