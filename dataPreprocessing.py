#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
geo ai test
"""


import polars as pl


TEST_IMAGE_PATH: str = "./images/6zM_LlnQo9JD9zALzWdKtw.jpeg"


def main() -> None:
    df = pl.read_csv("./images.csv")
    print(df.head(5))

    return


if __name__ == "__main__":
	main()

