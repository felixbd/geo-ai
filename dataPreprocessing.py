#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
geo ai test
"""

import polars as pl
from tqdm import tqdm
import os


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import reverse_geocoder as rg

import geopandas as gpd
from shapely.geometry import Point

OUT_CSV: str = "new-out.csv"

# Load the shapefiles (country and state borders)
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# states = gpd.read_file('path_to_your_states_shapefile.shp')  # Downloaded separately


def get_country_from_coordinates_old(lat: float, lng: float) -> str:
    geo_locator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geo_locator.reverse((lat, lng), language='en', exactly_one=True)
        if location and 'country' in location.raw['address']:
            return location.raw['address']['ISO3166-2-lvl4']  # ['country']
        else:
            return "Country not found"
    except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        return str(e)


def get_country_from_coordinates(lat: float, lng: float) -> str:
    try:
        results = rg.search((lat, lng))  # reverse geocoding
        if results and 'cc' in results[0]:
            return results[0]['cc']  # cc name
        else:
            return "none"
    except Exception as e:
        return f"ERROR: {e}"




def save_country_to_csv(df_in: pl.DataFrame, out_csv_name: str = OUT_CSV) -> None:
    rv_df = pl.DataFrame({
        "latitude": pl.Series([], dtype=pl.Float64),
        "longitude": pl.Series([], dtype=pl.Float64),
        "ISO3166-2-lvl4": pl.Series([], dtype=pl.Utf8),
    })

    # after overwriting half an hour of waiting, I decided to add a warning
    if os.path.exists(out_csv_name):
        print(f"Warning: {out_csv_name} already exists and will be overwritten")
        if input("Do you want to continue?\nThe existing file will be overwritten! [y/N]: ").lower() != "y":
            print("Exiting")
            return

    try:
        for i in tqdm(range(df_in.shape[0])):
            row = df_in.row(i)

            latitude, longitude = row[1:3]
            iso_code = get_country_from_coordinates(latitude, longitude)

            new_row = pl.DataFrame({
                "latitude": [latitude],
                "longitude": [longitude],
                "ISO3166-2-lvl4": [iso_code],
            })

            rv_df.extend(new_row)
    except KeyboardInterrupt as e:
        print("Interrupted by user")
        rv_df.write_csv(f"./{out_csv_name}")
        return

    rv_df.write_csv(out_csv_name)


def main() -> None:
    df = pl.read_csv("./images.csv")
    print(df.head(5))
    print(df.shape)

    save_country_to_csv(df)

    return


if __name__ == "__main__":
    main()
