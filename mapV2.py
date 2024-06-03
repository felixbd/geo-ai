import plotly.graph_objects as go
import geopandas as gpd
import plotly.express as px

# Create figure
fig = go.Figure()

# Add coastlines
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig.add_trace(go.Choroplethmapbox(geojson=world.__geo_interface__, z=[0]*len(world),
                                  colorscale="Viridis", showscale=False))

# Define the gridlines
latitudes = list(range(-90, 91))
longitudes = list(range(-180, 181))

# Add latitude lines
for lat in latitudes:
    fig.add_trace(go.Scattermapbox(
        lon=[-180, 180],
        lat=[lat, lat],
        mode='lines',
        line=dict(width=1, color='gray'),
        name=f'Lat {lat}'
    ))

# Add longitude lines
for lon in longitudes:
    fig.add_trace(go.Scattermapbox(
        lon=[lon, lon],
        lat=[-90, 90],
        mode='lines',
        line=dict(width=1, color='gray'),
        name=f'Lon {lon}'
    ))

# Set the layout for the map
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=0, lon=0),
        zoom=1
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

# Show the plot
fig.show()
