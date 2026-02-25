#!/usr/bin/env python3
"""
Random Forest model for handwritten digit classification.

- Train + Validation: dataset-digit/archive/{0..9}/  (~21,555 images, 90x140 JPG)
- Test:               dataset-digit/Test/{0..9}/      (~3,067 images, 128x128 PNG)

Pipeline: Grayscale -> Resize 28x28 -> Flatten (784) -> StandardScaler + PCA(95%) -> RandomForest
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.transform import resize
from scipy.ndimage import center_of_mass, shift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import joblib
import time

# ========================= CONFIG =========================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_VAL_DIR = BASE_DIR / "dataset-digit" / "archive"   # Kaggle dataset
TEST_DIR = BASE_DIR / "dataset-digit" / "Test"            # Custom test set
SAVE_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVE_DIR / "rf_digit_model.joblib"

IMG_SIZE = (28, 28)       # resize target
N_COMPONENTS = 0.95       # PCA: keep 95% variance
N_ESTIMATORS = 300        # Random Forest trees
RANDOM_STATE = 42
VAL_RATIO = 0.2           # 20% of archive data for validation
TEST_RATIO = 0.2          # 20% of archive data for internal test
# ==========================================================


def preprocess_digit(img):
    """
    MNIST-style preprocessing:
    Grayscale -> Otsu threshold -> Bounding box crop -> Resize 20x20
    -> Pad to 28x28 -> Center of mass shift.
    """
    # Grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        img = rgb2gray(img)

    # Normalize 0-1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Threshold (Otsu)
    thresh = threshold_otsu(img)
    img = img > thresh
    img = img.astype(np.float32)

    # Ensure digit is white
    if np.mean(img) > 0.5:
        img = 1 - img

    # Bounding box crop
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] == 0:
        return np.zeros((28, 28))  # tránh crash nếu ảnh trống

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = img[y_min:y_max+1, x_min:x_max+1]

    # Resize digit to 20x20
    digit = resize(digit, (20, 20), anti_aliasing=True)

    # Pad to 28x28
    canvas = np.zeros((28, 28))
    canvas[4:24, 4:24] = digit

    # Center shift (center of mass)
    cy, cx = center_of_mass(canvas)
    shift_y = 14 - cy
    shift_x = 14 - cx
    canvas = shift(canvas, (shift_y, shift_x))

    return canvas


def load_images_from_folder(folder: Path, img_size: tuple = IMG_SIZE):
    """
    Load all images from folder/{0..9}/, apply MNIST-style preprocessing,
    flatten to 1D vector.
    Returns (X, y) as numpy arrays.
    """
    images = []
    labels = []
    total = 0

    for digit in range(10):
        digit_dir = folder / str(digit)
        if not digit_dir.exists():
            print(f"  [WARNING] Folder not found: {digit_dir}")
            continue

        files = sorted(digit_dir.iterdir())
        count = 0
        for f in files:
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            try:
                img = np.array(Image.open(f))              # load raw
                processed = preprocess_digit(img)           # MNIST-style preprocess
                images.append(processed.flatten())          # (784,)
                labels.append(digit)
                count += 1
            except Exception as e:
                print(f"  [ERROR] Cannot read {f}: {e}")

        total += count
        print(f"  Digit {digit}: {count} images")

    print(f"  => Total: {total} images loaded from {folder.name}/\n")
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def evaluate_model(model, X, y_true, set_name: str):
    """Evaluate model and print metrics + confusion matrix."""
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    print("=" * 60)
    print(f"  {set_name} Results")
    print("=" * 60)
    print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(y_true, y_pred, digits=4,
                                target_names=[f"Digit {d}" for d in range(10)]))

    # Confusion Matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical")
    ax.set_title(f"Confusion Matrix - {set_name} (Acc={acc:.4f})")
    plt.tight_layout()
    plt.show()

    return acc


def main():
    print("=" * 60)
    print("  Random Forest - Handwritten Digit Classification")
    print("=" * 60)

    # -------- 1. Load all archive data (~20k images) --------
    print(f"\n[1] Loading ALL data from: {TRAIN_VAL_DIR}")
    X_all, y_all = load_images_from_folder(TRAIN_VAL_DIR)
    print(f"    Shape: X={X_all.shape}, y={y_all.shape}")

    # -------- 2. Split into Train / Validation / Test (60% / 20% / 20%) --------
    train_ratio = 1 - VAL_RATIO - TEST_RATIO
    print(f"\n[2] Splitting: Train {train_ratio*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")

    # First split: separate out internal test set
    X_trainval, X_internal_test, y_trainval, y_internal_test = train_test_split(
        X_all, y_all,
        test_size=TEST_RATIO,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    # Second split: separate train and validation from remaining data
    val_fraction = VAL_RATIO / (1 - TEST_RATIO)  # val ratio relative to trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    print(f"    Train:          {X_train.shape[0]} samples")
    print(f"    Validation:     {X_val.shape[0]} samples")
    print(f"    Internal Test:  {X_internal_test.shape[0]} samples")

    # -------- 3. Combine Train + Validation for training --------
    X_train_combined = np.concatenate([X_train, X_val], axis=0)
    y_train_combined = np.concatenate([y_train, y_val], axis=0)
    print(f"\n[3] Combined Train+Val for training: {X_train_combined.shape[0]} samples")

    # -------- 4. Build pipeline: Scaler + PCA + RF --------
    print(f"\n[4] Building pipeline: StandardScaler -> PCA({N_COMPONENTS}) -> RandomForest({N_ESTIMATORS} trees)")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )),
    ])

    # -------- 5. Train on Train + Val --------
    print("\n[5] Training Random Forest on Train+Val...")
    t0 = time.time()
    pipeline.fit(X_train_combined, y_train_combined)
    elapsed = time.time() - t0
    print(f"    Training completed in {elapsed:.1f}s")

    n_pca = pipeline.named_steps["pca"].n_components_
    explained = pipeline.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"    PCA: {784} -> {n_pca} dimensions (explained variance: {explained:.4f})")

    # -------- 6. Evaluate on Internal Test (same dataset) --------
    print(f"\n[6] Evaluating on INTERNAL TEST set ({X_internal_test.shape[0]} samples - same dataset)")
    X_int_transformed = pipeline.named_steps["scaler"].transform(X_internal_test)
    X_int_transformed = pipeline.named_steps["pca"].transform(X_int_transformed)
    internal_test_acc = evaluate_model(pipeline.named_steps["rf"], X_int_transformed, y_internal_test, "INTERNAL TEST (same dataset)")

    # -------- 7. Load & Evaluate on External Test set (dataset-digit/Test) --------
    print(f"\n[7] Loading EXTERNAL TEST data from: {TEST_DIR}")
    X_ext_test, y_ext_test = load_images_from_folder(TEST_DIR)
    print(f"    Shape: X={X_ext_test.shape}, y={y_ext_test.shape}")

    print(f"\n[8] Evaluating on EXTERNAL TEST set ({X_ext_test.shape[0]} samples)")
    X_ext_transformed = pipeline.named_steps["scaler"].transform(X_ext_test)
    X_ext_transformed = pipeline.named_steps["pca"].transform(X_ext_transformed)
    external_test_acc = evaluate_model(pipeline.named_steps["rf"], X_ext_transformed, y_ext_test, "EXTERNAL TEST (new data)")

    # -------- 8. Save model --------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[9] Model saved to: {MODEL_PATH}")

    # -------- Summary --------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total archive samples:      {X_all.shape[0]}")
    print(f"  Train samples:              {X_train.shape[0]}")
    print(f"  Validation samples:         {X_val.shape[0]}")
    print(f"  Train+Val (used to train):  {X_train_combined.shape[0]}")
    print(f"  Internal Test samples:      {X_internal_test.shape[0]}")
    print(f"  External Test samples:      {X_ext_test.shape[0]}")
    print(f"  PCA dimensions:             {n_pca}")
    print(f"  Internal Test Accuracy:     {internal_test_acc:.4f}")
    print(f"  External Test Accuracy:     {external_test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
