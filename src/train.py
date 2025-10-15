"""End-to-end training pipeline: generators, callbacks, 2-stage training, export."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import (
DATA_PARTS, CHECKPOINT_PATH, EXPORT_DIR, IMG_SIZE, BATCH_SIZE, SEED,
EPOCHS_STAGE1, EPOCHS_STAGE2, LR_STAGE2
)

from .data_loader import collect_images, build_dataframe, stratified_split
from .model_builder import build_model


# Optional: be polite with GPU memory
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass




def _make_gens(train_df, val_df, test_df):
    """Build Keras generators with augmentation for train and plain for val/test."""
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    plain = ImageDataGenerator(rescale=1./255)


    train_gen = train_aug.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath', y_col='age',
        target_size=IMG_SIZE, color_mode='rgb',
        batch_size=BATCH_SIZE, class_mode='raw',
        shuffle=True, seed=SEED,
    )
    
    val_gen = plain.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath', y_col='age',
        target_size=IMG_SIZE, color_mode='rgb',
        batch_size=BATCH_SIZE, class_mode='raw',
        shuffle=False,
    )
    
    test_gen = plain.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath', y_col='age',
        target_size=IMG_SIZE, color_mode='rgb',
        batch_size=BATCH_SIZE, class_mode='raw',
        shuffle=False,
    )
    
    return train_gen, val_gen, test_gen

def train_model():
    """Run the full pipeline and return (model, history, export_dir)."""
    # 1) Discover & label data
    files = collect_images(DATA_PARTS)
    if not files:
        raise RuntimeError("No images found. Put data in data/part1..3 with UTKFace-style names.")
    df = build_dataframe(files)


    # 2) Split
    train_df, val_df, test_df = stratified_split(df)
    print(f"Split sizes → train={len(train_df)} val={len(val_df)} test={len(test_df)}")


    # 3) Generators
    train_gen, val_gen, test_gen = _make_gens(train_df, val_df, test_df)


    # 4) Model + callbacks
    model = build_model()
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_PATH), monitor='val_loss',
        save_best_only=True, save_weights_only=True, verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=5, verbose=1
    )
    
    # 5) Stage-1: head only
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1,
    )
    
    # 6) Stage-2: unfreeze tail (~last 50 layers) and fine-tune with a smaller LR
    for layer in model.layers[0].layers[:-50]:
        layer.trainable = False
    for layer in model.layers[0].layers[-50:]:
        layer.trainable = True


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE2),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE2,
        initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1,
    )


    # 7) Evaluate
    test_metrics = model.evaluate(test_gen, steps=len(test_gen), verbose=1)
    print("Test set evaluation (loss, mae):", test_metrics)


    # 8) Export: weights + SavedModel with explicit signature
    export_dir = Path(EXPORT_DIR) / datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir.mkdir(parents=True, exist_ok=True)


    model.save_weights(str(export_dir/"final.weights.h5"))


    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE[0], IMG_SIZE[1], 3], tf.float32)])
    def serve(x):
        y = model(x, training=False)
        return {"age": y}


    tf.saved_model.save(model, str(export_dir), signatures={"serving_default": serve})
    print("SavedModel exported to:", str(export_dir))


    # Return for optional downstream use
    class _History:
        # small wrapper so you can access both if you want
        h1, h2 = history1, history2
    return model, _History(), str(export_dir)