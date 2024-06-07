#!/usr/bin/env python3

"""
geo ai
"""

import polars as pl

if 0:
	import tensorflow as tf
	from tensorflow.keras.models import Model
	from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	import numpy as np


X_SCALE: int = 10  # 10  # 1000
Y_SCALE: int = 10  # 20 # 2000


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


def main() -> None:
	clean_data_frame()


if __name__ == "__main__":
	main()
