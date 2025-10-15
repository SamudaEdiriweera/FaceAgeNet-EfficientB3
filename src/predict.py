"""Single-image inference for SavedModel or live Keras model."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from .config import IMG_SIZE




def _load_img(path: str | Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = (np.array(im, dtype=np.float32) / 255.0)[None, ...]
    return arr




def predict_with_savedmodel(export_dir: str | Path, img_path: str | Path) -> float:
    m = tf.saved_model.load(str(export_dir))
    f = m.signatures["serving_default"]
    x = _load_img(img_path)
    y = f(tf.constant(x))["age"].numpy().squeeze().item()
    return float(y)




def predict_with_keras(model: tf.keras.Model, img_path: str | Path) -> float:
    x = _load_img(img_path)
    y = model.predict(x, verbose=0)[0][0]
    return float(y)