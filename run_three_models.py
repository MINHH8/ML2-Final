#!/usr/bin/env python3
"""Benchmark the 3 project models: SVM, RandomForest, CNN."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

from ml2_data_utils import (
    compute_classification_metrics,
    has_labeled_digit_folders,
    load_images_from_label_folders,
    load_train_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVM + RF + CNN with unified split")
    parser.add_argument("--train_dir", type=Path, default=Path("dataset-digit/archive"))
    parser.add_argument("--test_dir", type=Path, default=Path("dataset-digit/Test"))
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--n_components", type=float, default=0.95)
    parser.add_argument("--cnn_epochs", type=int, default=15)
    parser.add_argument("--cnn_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, default=Path("results/chapter4/three_models"))
    parser.add_argument(
        "--mnist_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MNIST fallback if --train_dir does not exist.",
    )
    parser.add_argument(
        "--use_external_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate on --test_dir if labeled folders are available.",
    )
    return parser.parse_args()


def evaluate_predictions(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    inference_seconds: float,
) -> Dict[str, float]:
    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "model": model_name,
        "split": split_name,
        "n_samples": int(len(y_true)),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        **metrics,
    }


def extract_hog_features(X_flat: np.ndarray) -> np.ndarray:
    """Extract HOG features from flattened 28x28 images."""
    X_img = X_flat.reshape(-1, 28, 28)
    feats = [
        hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        for img in X_img
    ]
    return np.asarray(feats, dtype=np.float32)


def build_cnn_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if min(args.train_ratio, args.val_ratio, args.test_ratio) <= 0:
        raise ValueError("All split ratios must be > 0.")
    if not np.isclose(ratio_sum, 1.0, atol=1e-8):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got train+val+test={ratio_sum:.6f}."
        )

    print("=" * 72)
    print("Chapter 4 - Project Models (SVM + RandomForest + CNN)")
    print("=" * 72)

    X_all, y_all, source = load_train_dataset(
        train_dir=args.train_dir,
        allow_mnist_fallback=args.mnist_fallback,
        max_samples=args.max_train_samples,
        random_state=args.random_state,
        verbose=True,
    )
    print(f"[INFO] Training source: {source}")
    print(f"[INFO] X shape={X_all.shape}, y shape={y_all.shape}")

    X_trainval, X_internal_test, y_trainval, y_internal_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_ratio,
        stratify=y_all,
        random_state=args.random_state,
    )
    val_fraction = args.val_ratio / (args.train_ratio + args.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=args.random_state,
    )
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    print("[INFO] Split summary:")
    print(
        f"  Ratio (train/val/test): "
        f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}"
    )
    print(f"  Train:         {len(y_train)}")
    print(f"  Validation:    {len(y_val)}")
    print(f"  Internal test: {len(y_internal_test)}")
    print(f"  Fit (Train+Val): {len(y_fit)}")

    X_external: Optional[np.ndarray] = None
    y_external: Optional[np.ndarray] = None
    if args.use_external_test and has_labeled_digit_folders(args.test_dir):
        print(f"[INFO] Loading external test from: {args.test_dir}")
        X_external, y_external, _ = load_images_from_label_folders(args.test_dir, verbose=True)
        print(f"  External test shape={X_external.shape}")
    else:
        print("[INFO] External test skipped (folder missing or disabled).")

    rows = []

    # ---------------------- RandomForest ----------------------
    print("\n[INFO] Training model: RandomForest")
    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=args.n_components, random_state=args.random_state)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    rf_pipeline.fit(X_fit, y_fit)
    rf_train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_internal = rf_pipeline.predict(X_internal_test)
    rf_internal_infer = time.perf_counter() - t0
    rows.append(
        evaluate_predictions(
            "RandomForest",
            "internal_test",
            y_internal_test,
            y_pred_internal,
            rf_train_sec,
            rf_internal_infer,
        )
    )

    if X_external is not None and y_external is not None and len(y_external) > 0:
        t0 = time.perf_counter()
        y_pred_external = rf_pipeline.predict(X_external)
        rf_external_infer = time.perf_counter() - t0
        rows.append(
            evaluate_predictions(
                "RandomForest",
                "external_test",
                y_external,
                y_pred_external,
                rf_train_sec,
                rf_external_infer,
            )
        )

    # ---------------------- SVM (HOG) ----------------------
    print("\n[INFO] Training model: SVM(HOG)")
    X_fit_hog = extract_hog_features(X_fit)
    X_internal_hog = extract_hog_features(X_internal_test)
    X_external_hog = extract_hog_features(X_external) if X_external is not None else None

    svm_scaler = StandardScaler()
    X_fit_hog_scaled = svm_scaler.fit_transform(X_fit_hog)
    X_internal_hog_scaled = svm_scaler.transform(X_internal_hog)
    X_external_hog_scaled = svm_scaler.transform(X_external_hog) if X_external_hog is not None else None

    svm = LinearSVC(
        C=1.0,
        random_state=args.random_state,
        max_iter=20000,
    )
    t0 = time.perf_counter()
    svm.fit(X_fit_hog_scaled, y_fit)
    svm_train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_internal = svm.predict(X_internal_hog_scaled)
    svm_internal_infer = time.perf_counter() - t0
    rows.append(
        evaluate_predictions(
            "SVM(HOG)",
            "internal_test",
            y_internal_test,
            y_pred_internal,
            svm_train_sec,
            svm_internal_infer,
        )
    )

    if X_external_hog_scaled is not None and y_external is not None and len(y_external) > 0:
        t0 = time.perf_counter()
        y_pred_external = svm.predict(X_external_hog_scaled)
        svm_external_infer = time.perf_counter() - t0
        rows.append(
            evaluate_predictions(
                "SVM(HOG)",
                "external_test",
                y_external,
                y_pred_external,
                svm_train_sec,
                svm_external_infer,
            )
        )

    # ---------------------- CNN ----------------------
    print("\n[INFO] Training model: CNN")
    from tensorflow import keras

    keras.utils.set_random_seed(args.random_state)
    cnn = build_cnn_model()

    X_fit_cnn = X_fit.reshape(-1, 28, 28, 1).astype(np.float32)
    X_internal_cnn = X_internal_test.reshape(-1, 28, 28, 1).astype(np.float32)
    X_external_cnn = (
        X_external.reshape(-1, 28, 28, 1).astype(np.float32) if X_external is not None else None
    )

    t0 = time.perf_counter()
    cnn.fit(
        X_fit_cnn,
        y_fit,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        verbose=0,
    )
    cnn_train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_internal = np.argmax(cnn.predict(X_internal_cnn, verbose=0), axis=1)
    cnn_internal_infer = time.perf_counter() - t0
    rows.append(
        evaluate_predictions(
            "CNN",
            "internal_test",
            y_internal_test,
            y_pred_internal,
            cnn_train_sec,
            cnn_internal_infer,
        )
    )

    if X_external_cnn is not None and y_external is not None and len(y_external) > 0:
        t0 = time.perf_counter()
        y_pred_external = np.argmax(cnn.predict(X_external_cnn, verbose=0), axis=1)
        cnn_external_infer = time.perf_counter() - t0
        rows.append(
            evaluate_predictions(
                "CNN",
                "external_test",
                y_external,
                y_pred_external,
                cnn_train_sec,
                cnn_external_infer,
            )
        )

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "three_models_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n[RESULT] 3-model metrics:")
    print(df.to_string(index=False))
    print(f"\n[OK] Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
