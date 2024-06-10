#!/usr/bin/env python3

"""
geo ai
"""


import polars as pl
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error, \
    mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, \
    mean_absolute_percentage_error

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score, roc_auc_score, roc_curve, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef


X_SCALE: int = 10  # 1000
Y_SCALE: int = 20  # 2000

target_width: int  = 512
target_height: int = 256

MODEL_PATH: str = "./my_model_new.keras"


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


def ids_to_gcs(x: float, y: float) -> tuple[float, float]:
    """ inverse of gcs_to_ids """
    return id_to_lat(round(x)), id_to_lon(round(y))


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
    output_1 = Dense(1, activation='linear', name='output_1')(dense_layer)
    output_2 = Dense(1, activation='linear', name='output_2')(dense_layer)

    model = Model(inputs=input_layer, outputs=[output_1, output_2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 'adam'
                  loss={'output_1': 'mean_squared_error', 'output_2': 'mean_squared_error'},
                  metrics={'output_1': 'mean_absolute_error', 'output_2': 'mean_absolute_error'})

    return model


def train_model(model: tf.keras.models.Model) -> None:
    rv: list = []

    df = pl.read_csv("./new-images.csv")

    temp = 1_500  # 1_000  # df.shape[0]

    for i in tqdm(range(temp), desc="Training model (looping over images)"):
        row = df.row(i)
        lat, lon = row[1], row[2]
        image_path = f"./images/{row[0]}.jpeg"

        # rv.append([lat, lon, image_path])

        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (target_height, target_width))
        image = image / 255.0

        rv.append([lat, lon, image])

        # Train the model
        # model.fit(image, {'output_1': lat, 'output_2': lon})  # , epochs=50, batch_size=32)  # , validation_split=0.2)

    lats = np.array([e[0] for e in rv])
    lons = np.array([e[1] for e in rv])
    imgs = np.array([e[2] for e in rv])

    # model.fit(imgs, {'output_1': lats, 'output_2': lons}, epochs=50, batch_size=32, validation_split=0.2)
    model.fit(imgs, {'output_1': lats, 'output_2': lons}, epochs=10, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)  # or `keras.saving.save_model(model, MODEL_PATH)

    return


def load_and_predict() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rv: list[tuple[tuple[float, float], tuple[float, float]]] = []

    model = tf.keras.models.load_model(MODEL_PATH)
    df = pl.read_csv("./new-images.csv")

    df = df.head(1500)
    samples = df.sample(10)

    for i in range(10):
        row = samples.row(i)
        lat, lon = row[1], row[2]
        image_path = f"./images/{row[0]}.jpeg"

        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (target_height, target_width))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = model.predict(image)

        rv.append(((lat, lon), (prediction[0][0][0], prediction[1][0][0])))

    return rv


def main() -> None:
    # clean_data_frame()

    model = generate_model()
    train_model(model)

    """
    true_vs_pred = load_and_predict()
    true_vs_pred_gcs = [(ids_to_gcs(*true), ids_to_gcs(*pred)) for true, pred in true_vs_pred]
    print(f"{ true_vs_pred = }\n{ true_vs_pred_gcs = }")

    true_coords = [true for true, pred in true_vs_pred]
    pred_coords = [pred for true, pred in true_vs_pred]
    true_coords_flat = [coord for pair in true_coords for coord in pair]
    pred_coords_flat = [coord for pair in pred_coords for coord in pair]
    mse = mean_squared_error(true_coords_flat, pred_coords_flat)
    print(f"Mean squared error: {mse}")

    # further metrics
    mae = mean_absolute_error(true_coords_flat, pred_coords_flat)
    print(f"Mean absolute error: {mae}")
    r2 = r2_score(true_coords_flat, pred_coords_flat)
    print(f"R2 score: {r2}")
    explained_variance = explained_variance_score(true_coords_flat, pred_coords_flat)
    print(f"Explained variance score: {explained_variance}")
    max_err = max_error(true_coords_flat, pred_coords_flat)
    print(f"Max error: {max_err}")
    mse_log = mean_squared_log_error(true_coords_flat, pred_coords_flat)
    print(f"Mean squared log error: {mse_log}")
    median_ae = median_absolute_error(true_coords_flat, pred_coords_flat)
    print(f"Median absolute error: {median_ae}")
    mean_poisson_dev = mean_poisson_deviance(true_coords_flat, pred_coords_flat)
    print(f"Mean Poisson deviance: {mean_poisson_dev}")
    """


if __name__ == "__main__":
    main()

