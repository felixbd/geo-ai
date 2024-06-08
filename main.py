#!/usr/bin/env python3

"""
geo ai
"""

import polars as pl
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

X_SCALE: int = 10  # 10  # 1000
Y_SCALE: int = 10  # 20 # 2000

target_width = 512
target_height = 256  # Define a fixed height


def lat_to_id(lat: float) -> int:
    return round((lat if lat >= 0 else (lat * (-1)) + 90) * X_SCALE / 180)


def lon_to_id(lon: float) -> int:
    return round((lon if lon >= 0 else (lon * (-1)) + 180) * Y_SCALE / 360)


def gcs_to_ids(a: float, b: float) -> tuple[int, int]:
    """ geographic coordinate system (GCS) -> (int, int) """
    return lat_to_id(a), lon_to_id(b)


def id_to_lat(x: int) -> float:
    rv = x * 180 / X_SCALE
    return rv if rv <= 90 else (rv - 90) * -1


def id_to_lon(y: int) -> float:
    rv = y * 360 / Y_SCALE
    return rv if rv <= 180 else (rv - 180) * -1


def ids_to_gcs(x: int, y: int) -> tuple[float, float]:
    """ inverse of gcs_to_ids """
    return id_to_lat(x), id_to_lon(y)


def clean_data_frame() -> None:
    df = pl.read_csv("./images.csv")

    # map the latitude and longitude to the ids
    df = df.with_columns(
        pl.col("lat").map_elements(lambda a: lat_to_id(a), return_dtype=pl.UInt16),
        pl.col("lng").map_elements(lambda b: lon_to_id(b), return_dtype=pl.UInt16),
    )

    df.write_csv("./new-images.csv")


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------
#
#	deep learning section
#
# ----------


"""
# Train the model
history = model.fit(X_train, {'output_1': Y_train[:, 0], 'output_2': Y_train[:, 1]}, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2)

# Make predictions
Y_pred = model.predict(X_test)
y_pred_1 = Y_pred[0]
y_pred_2 = Y_pred[1]

# Evaluate the model
accuracy_1 = np.mean((y_pred_1 > 0.5).astype(int) == Y_test[:, 0])
accuracy_2 = np.mean((y_pred_2 > 0.5).astype(int) == Y_test[:, 1])

print(f"Accuracy for the first label set: {accuracy_1}")
print(f"Accuracy for the second label set: {accuracy_2}")
"""


def generate_model() -> tf.keras.models.Model:
    # maybe use gelu activation function instead of relu

    input_layer = Input(shape=(target_height, target_width, 3))

    # Convolutional layers
    conv_layer = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)

    conv_layer = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)

    conv_layer = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)

    conv_layer = Flatten()(conv_layer)

    # Fully connected layers
    dense_layer = Dense(128, activation='relu')(conv_layer)
    dense_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)

    # Output layers for y1 and y2
    output_1 = Dense(1, activation='sigmoid', name='output_1')(dense_layer)  # TODO: adapt activation function
    output_2 = Dense(1, activation='sigmoid', name='output_2')(dense_layer)  # TODO: adapt activation function

    model = Model(inputs=input_layer, outputs=[output_1, output_2])
    model.compile(optimizer="adam",
                  loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy'},
                  metrics={'output_1': 'accuracy', 'output_2': 'accuracy'})

    return model


def train_model(model: tf.keras.models.Model) -> None:
    df = pl.read_csv("./new-images.csv")

    temp = 5  # df.shape[0]

    for i in tqdm(range(temp), desc="Training model (looping over images)"):
        row = df.row(i)
        lat, lon = row[1], row[2]
        image_path = f"./images/{row[0]}.jpeg"

        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (target_height, target_width))
        image = image / 255.0

        # Load the labels
        label_1 = lat
        label_2 = lon

        # Train the model
        model.fit(image, {'output_1': label_1, 'output_2': label_2}, epochs=50, batch_size=32)  # , validation_split=0.2)

    # save the model
    model.save("model.h5")


def main() -> None:
    # clean_data_frame()
    model = generate_model()
    train_model(model)


if __name__ == "__main__":
    main()
