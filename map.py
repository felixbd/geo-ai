#!/usr/bin/env python3

"""
world map with grid every 0.5 degrees (latitude, longitude)
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


ROBINSON: bool = True
XR: int = 10
YR: int = 10

def main() -> None:
    # Create a figure and an axes with a specific projection
    fig = plt.figure(figsize=(15, 7))
    if not ROBINSON:
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = plt.axes(projection=ccrs.Robinson())


    # Add features to the map (e.g., coastlines, borders)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    # Draw gridlines every 0.5 degrees
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(range(-180, 181, 1))
    gl.ylocator = plt.FixedLocator(range(-90, 91, 1))

    # Add gridlines every 0.5 degrees
    for lat in range(-90, 91, XR):
        ax.plot([-180, 180], [lat, lat], color='red', linestyle='--', linewidth=2, transform=ccrs.PlateCarree())
    for lon in range(-180, 181, YR):
    	ax.plot([lon, lon], [-90, 90], color='green', linestyle='--', linewidth=2, transform=ccrs.PlateCarree())

    # Set the extent of the map to the world
    # ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Show the plot
    plt.show()


if __name__ == "__main__":
	main()
