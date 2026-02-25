#!/usr/bin/env python3
"""Run clustering experiments for Chapter 4.1."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ml2_data_utils import cluster_purity, load_train_dataset


def compute_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int) -> float:
    """Safe silhouette computation with optional sampling."""
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return float("nan")

    sample = min(sample_size, len(X)) if sample_size > 0 else len(X)
    return float(
        silhouette_score(
            X,
            labels,
            sample_size=sample,
            random_state=random_state,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clustering benchmark for handwritten digits")
    parser.add_argument("--train_dir", type=Path, default=Path("dataset-digit/archive"))
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_components", type=float, default=0.95, help="PCA components or explained variance")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--silhouette_sample_size", type=int, default=5000)
    parser.add_argument("--output_dir", type=Path, default=Path("results/chapter4/clustering"))
    parser.add_argument(
        "--mnist_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MNIST fallback if --train_dir does not exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chapter 4.1 - Clustering")
    print("=" * 72)

    X, y, source = load_train_dataset(
        train_dir=args.train_dir,
        allow_mnist_fallback=args.mnist_fallback,
        max_samples=args.max_train_samples,
        random_state=args.random_state,
        verbose=True,
    )
    print(f"[INFO] Data source: {source}")
    print(f"[INFO] X shape={X.shape}, y shape={y.shape}")

    print("[INFO] Running StandardScaler + PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    pca = PCA(n_components=args.n_components, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] PCA output shape={X_pca.shape}, explained_variance={explained:.4f}")

    records = []

    print("[INFO] Running KMeans...")
    t0 = time.perf_counter()
    km = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=20,
    )
    km_labels = km.fit_predict(X_pca)
    km_train_sec = time.perf_counter() - t0

    records.append(
        {
            "method": "KMeans",
            "n_samples": len(y),
            "n_features_after_pca": X_pca.shape[1],
            "explained_variance": explained,
            "train_seconds": km_train_sec,
            "silhouette": compute_silhouette(
                X_pca, km_labels, args.silhouette_sample_size, args.random_state
            ),
            "ari": float(adjusted_rand_score(y, km_labels)),
            "nmi": float(normalized_mutual_info_score(y, km_labels)),
            "purity": cluster_purity(y, km_labels),
            "inertia": float(km.inertia_),
            "aic": float("nan"),
            "bic": float("nan"),
        }
    )

    print("[INFO] Running GaussianMixture...")
    t0 = time.perf_counter()
    gmm = GaussianMixture(
        n_components=args.n_clusters,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=args.random_state,
    )
    try:
        gmm_labels = gmm.fit_predict(X_pca)
        gmm_name = "GaussianMixture(full)"
    except ValueError:
        # Fallback for unstable full-covariance runs on small/degenerate subsets.
        gmm = GaussianMixture(
            n_components=args.n_clusters,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=args.random_state,
        )
        gmm_labels = gmm.fit_predict(X_pca)
        gmm_name = "GaussianMixture(diag)"
    gmm_train_sec = time.perf_counter() - t0

    records.append(
        {
            "method": gmm_name,
            "n_samples": len(y),
            "n_features_after_pca": X_pca.shape[1],
            "explained_variance": explained,
            "train_seconds": gmm_train_sec,
            "silhouette": compute_silhouette(
                X_pca, gmm_labels, args.silhouette_sample_size, args.random_state
            ),
            "ari": float(adjusted_rand_score(y, gmm_labels)),
            "nmi": float(normalized_mutual_info_score(y, gmm_labels)),
            "purity": cluster_purity(y, gmm_labels),
            "inertia": float("nan"),
            "aic": float(gmm.aic(X_pca)),
            "bic": float(gmm.bic(X_pca)),
        }
    )

    df = pd.DataFrame(records)
    metrics_path = args.output_dir / "clustering_metrics.csv"
    df.to_csv(metrics_path, index=False)

    print("\n[RESULT] Clustering metrics:")
    print(df.to_string(index=False))
    print(f"\n[OK] Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
