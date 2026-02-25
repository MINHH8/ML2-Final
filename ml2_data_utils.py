#!/usr/bin/env python3
"""Shared data utilities for Chapter 4 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def has_labeled_digit_folders(root_dir: Path) -> bool:
    """Return True when at least one digit folder (0..9) exists."""
    if not root_dir.exists():
        return False
    return any((root_dir / str(d)).exists() for d in range(10))


def preprocess_digit(img: np.ndarray) -> np.ndarray:
    """
    MNIST-style preprocessing:
    grayscale -> Otsu threshold -> crop -> resize 20x20 -> pad 28x28 -> center shift.
    """
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        img = rgb2gray(img)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    thresh = threshold_otsu(img)
    img = (img > thresh).astype(np.float32)

    if np.mean(img) > 0.5:
        img = 1 - img

    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    digit = img[y_min : y_max + 1, x_min : x_max + 1]

    digit = resize(digit, (20, 20), anti_aliasing=True)

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = digit

    cy, cx = center_of_mass(canvas)
    shift_y = 14 - cy
    shift_x = 14 - cx
    canvas = shift(canvas, (shift_y, shift_x))
    return canvas.astype(np.float32)


def load_images_from_label_folders(
    root_dir: Path,
    max_per_digit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Load images from root_dir/{0..9}/.
    Returns:
      X: (n_samples, 784) float32
      y: (n_samples,) int32
      stats: dict[digit] -> count
    """
    images = []
    labels = []
    stats: Dict[int, int] = {}

    for digit in range(10):
        digit_dir = root_dir / str(digit)
        if not digit_dir.exists():
            continue

        files = [
            p
            for p in sorted(digit_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_per_digit is not None:
            files = files[:max_per_digit]

        count = 0
        for f in files:
            try:
                img = np.array(Image.open(f))
                processed = preprocess_digit(img)
                images.append(processed.flatten())
                labels.append(digit)
                count += 1
            except Exception as ex:  # pragma: no cover - best-effort logging
                if verbose:
                    print(f"[WARN] Skip unreadable image: {f} ({ex})")

        stats[digit] = count
        if verbose:
            print(f"  Digit {digit}: {count} images")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    if verbose:
        print(f"  Total loaded from {root_dir}: {len(y)}")
    return X, y, stats


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep at most max_samples samples with label stratification."""
    if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
        return X, y

    X_keep, _, y_keep, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_keep.astype(np.float32), y_keep.astype(np.int32)


def load_train_dataset(
    train_dir: Path,
    allow_mnist_fallback: bool = True,
    max_samples: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load train dataset.
    Priority:
      1) local labeled folders from train_dir/{0..9}
      2) MNIST fallback (tensorflow.keras.datasets.mnist)
    """
    if has_labeled_digit_folders(train_dir):
        if verbose:
            print(f"[INFO] Training source: local folder -> {train_dir}")
        X, y, _ = load_images_from_label_folders(train_dir, verbose=verbose)
        X, y = stratified_subsample(X, y, max_samples=max_samples, random_state=random_state)
        return X, y, "local-folder"

    if not allow_mnist_fallback:
        raise FileNotFoundError(
            f"Training folder not found or empty: {train_dir}. "
            "Enable MNIST fallback or provide a valid --train_dir."
        )

    if verbose:
        print("[INFO] Training source: TensorFlow MNIST fallback")

    from tensorflow.keras.datasets import mnist

    (x_train, y_train), _ = mnist.load_data()
    X = (x_train.astype(np.float32) / 255.0).reshape(len(y_train), -1)
    y = y_train.astype(np.int32)
    X, y = stratified_subsample(X, y, max_samples=max_samples, random_state=random_state)
    if verbose:
        print(f"  Total loaded from MNIST: {len(y)}")
    return X, y, "mnist-fallback"


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy + macro precision/recall/F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def cluster_purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster purity score in [0,1] computed by majority label per cluster."""
    total = 0
    for c in np.unique(clusters):
        mask = clusters == c
        if not np.any(mask):
            continue
        counts = np.bincount(y_true[mask], minlength=10)
        total += int(counts.max())
    return float(total / len(y_true))
