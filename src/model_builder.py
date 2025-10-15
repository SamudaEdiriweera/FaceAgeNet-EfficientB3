"""EfficientNetB3 backbone + small regression head (transfer learning)."""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from .config import IMG_SIZE, LR_STAGE1




def build_model():
    """Create and compile a Keras model for age regression using EfficientNetB3 features."""
    base = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling='fromavg', # global average pooling gives a 1D feature vector
    )
    base.trainable = False # stage-1: freeze backbone


    model = Sequential([
        base,
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='linear'), # regression output = predicted age
    ])


    model.compile(
        optimizer=Adam(learning_rate=LR_STAGE1),
        loss=tf.keras.losses.Huber(), # robust regression
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )
    return model