#!/usr/bin/env python3

"""
convert

	lat [-90, 90], long [-180, 180]
->
	int [0, n], int [0, m]

and back
"""

import latexify

xs = [
	(34.0522, -118.2437),  # Los Angeles
	(-33.8688, 151.2093),  # Sydney
	(52.5200, 13.4050),    # Berlin
	(40.7128, -74.0060)    # New York City
]

X_SCALE: int = 1000
Y_SCALE: int = 2000

def f(a: float, b: float) -> (int, int):
	a = a if a >= 0 else (a * (-1)) + 90
	b = b if b >= 0 else (b * (-1)) + 180
	return round(a * X_SCALE / 180), round(b * Y_SCALE / 360)


def inverse_f(x: int, y: int) -> (float, float):
    # Undo the scaling
    a = x * 180 / X_SCALE
    b = y * 360 / Y_SCALE

    # Undo the negative adjustment
    # NOTE: is a bit curst for latexify
    a = a if a <= 90 else (a - 90) * -1
    b = b if b <= 180 else (b - 180) * -1
    # if a > 90: a = (a - 90) * -1
    # if b > 180: b = (b - 180) * -1

    return a, b


if __name__ == "__main__":
	print(f"{ xs[0] = }")
	temp = f(xs[0][0], xs[0][1])
	print(f"{ inverse_f(temp[0], temp[1]) }")

	print("= " * 40)

	print(f"{ latexify.get_latex(f, reduce_assignments=True) = }\n\n")
	print(f"{ latexify.get_latex(inverse_f, reduce_assignments=True) = }")

