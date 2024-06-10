#!/usr/bin/env python3

"""
world map with grid every `n` degrees (latitude, longitude)
"""

import matplotlib.pyplot as plt
# plt.style.use(['dark_background'])  # , 'presentation'])
plt.style.use('ggplot')

import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["keymap.quit"] = ["q", "escape"]

ROBINSON: bool = True
XR: int = 10  # 18
YR: int = 10  # 36


TRUE_VS_PRED_GCS: list = [((36.0, 18.0), (18.0, 36.0)), ((90.0, 54.0), (36.0, 36.0)), ((36.0, 144.0), (36.0, 54.0)),
                          ((90.0, 54.0), (36.0, 36.0)), ((90.0, 54.0), (72.0, 0.0)), ((-36.0, 72.0), (36.0, 54.0)),
                          ((36.0, 0.0), (36.0, 36.0)), ((54.0, 18.0), (36.0, 36.0)), ((18.0, 54.0), (36.0, 36.0)),
                          ((18.0, 54.0), (36.0, 36.0))]


def main() -> None:
    # Create a figure and an axes with a specific projection
    fig = plt.figure(figsize=(15, 7))
    if not ROBINSON:
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = plt.axes(projection=ccrs.Robinson())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(range(-180, 181, 5))
    gl.ylocator = plt.FixedLocator(range(-90, 91, 5))

    lw: float = 1.5

    # Add gridlines every 0.5 degrees
    for lat in range(-90, 91, XR):
        if lat == 0:
            ax.plot([-180, 180], [lat, lat], color='blue', linestyle='-', linewidth=lw, transform=ccrs.PlateCarree())
        else:
            ax.plot([-180, 180], [lat, lat], color='red', linestyle='--', linewidth=lw, transform=ccrs.PlateCarree())
    for lon in range(-180, 181, YR):
        if lon == 0:
            ax.plot([lon, lon], [-90, 90], color='blue', linestyle='-', linewidth=lw, transform=ccrs.PlateCarree())
        else:
            ax.plot([lon, lon], [-90, 90], color='green', linestyle='--', linewidth=lw, transform=ccrs.PlateCarree())

    # plot true vs predicted GCS (with label as number)
    s = ['o', 'x', 's', '^', 'v', '<', '>', 'd', 'p', 'h']
    for i, (true, pred) in enumerate(TRUE_VS_PRED_GCS[:11]):
        ax.plot(true[1], true[0], f'b{s[i]}', markersize=10, transform=ccrs.PlateCarree(), label=f"True {i}")
        ax.plot(pred[1], pred[0], f'r{s[i]}', markersize=10, transform=ccrs.PlateCarree(), label=f"Pred {i}")

    plt.show()


if __name__ == "__main__":
    main()
