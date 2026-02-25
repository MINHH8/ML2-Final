import joblib
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# all img in folder 

MODEL_PATH = r"D:\ML2-Final\saved_models\rf_digit_model.joblib"
TEST_DIR   = r"D:\ML2-Final\mnist_test_png"

model = joblib.load(MODEL_PATH)

def preprocess_to_mnist_vector(img_path: str) -> np.ndarray:
    """
    Returns shape (1, 784) float32, in 0..255 scale (matching your training input).
    """
    img = Image.open(img_path).convert("L")          # grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float32)            # PIL -> numpy 

    # Optional auto-invert:
    # MNIST digits are bright on dark background; if your image is dark-on-white, invert it.
    # Heuristic: if background is mostly bright, invert.
    if arr.mean() > 127:
        arr = 255.0 - arr

    x = arr.reshape(1, -1)                           # (1, 784)
    return x

test_dir = Path(TEST_DIR)
image_paths = sorted(
    [p for p in test_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
)

for p in image_paths:
    x = preprocess_to_mnist_vector(str(p))
    pred = model.predict(x)[0]
    print(f"{p.name} -> predicted: {pred}")
