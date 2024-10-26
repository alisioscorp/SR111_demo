import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

lat=41.705881
lon=-3.797353

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [lat, lon], columns=["lat", "lon"])  #near Lima
df2 = pd.DataFrame(
    np.random.randn(1000, 2) / [15, 35] + [lat, lon], columns=["lat", "lon"])
df3 = pd.DataFrame(
    np.random.randn(1000, 2) / [15, 15] + [lat, lon], columns=["lat", "lon"])
df4 = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 60] + [lat, lon], columns=["lat", "lon"])

# list of data frames
dataframes = [df, df2, df3, df4] 

# dictionary to save data frames
frames={} 

for key, value in enumerate(dataframes):    
  frames[key] = value # assigning data frame from list to key in dictionary
  print("key: ", key)
  print(frames[key], "\n")
    
mapstyle = st.sidebar.selectbox(
    "Choose Map Style:",
    options=["light"], #"dark", "Satellite", "road"],
    format_func=str.capitalize,
)
# # Create a map layer with the given coordinates
# layer1 = pdk.Layer(type = 'ScatterplotLayer', # layer type
#                   data=df_bos, # data source
#                   get_position='[lon, lat]', # coordinates
#                   get_radius=500, # scatter radius
#                   get_color=[0,0,255],   # scatter color
#                   pickable=True # work with tooltip
#                   )

# # Can create multiple layers in a map
# # For more layer information
# # https://deckgl.readthedocs.io/en/latest/layer.html
# # Line layer https://pydeck.gl/gallery/line_layer.html
# layer2 = pdk.Layer('ScatterplotLayer',
#                   data=df_bos,
#                   get_position='[lon, lat]',
#                   get_radius=100,
#                   get_color=[255,0,255],
#                   pickable=True
#                   )

# >>> layer = pydeck.Layer(
# >>>     'HexagonLayer',
# >>>     UK_ACCIDENTS_DATA,
# >>>     get_position=['lng', 'lat'],
# >>>     auto_highlight=True,
# >>>     elevation_scale=50,
# >>>     pickable=True,
# >>>     elevation_range=[0, 3000],
# >>>     extruded=True,
# >>>     coverage=1)

st.pydeck_chart(
    pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-v9', #f"{mapstyle}",  # 'light', 'dark', 'mapbox://styles/mapbox/satellite-streets-v12', 'road'
        #layers=[layer1,layer2], # The following layer would be on top of the previous layers
        initial_view_state=pdk.ViewState(
#            latitude=-12.04,
#            longitude=-76.94,
            latitude=lat,
            longitude=lon,
            zoom=15,
            pitch=40, #50,
        ),
#     opacity=0.9,
#     get_position=["lng", "lat"],
#     threshold=0.75,
#     aggregation=pdk.types.String("MEAN"),
#     get_weight="weight",
#     pickable=True,
        layers=[
            pdk.Layer(
                "ScatterplotLayer", #"HeatmapLayer", #'HexagonLayer', #"ScatterplotLayer",
                data=df3,
                opacity=0.9,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                elevation_scale=10, 
                #get_radius=20,
                #elevation_range=[0, 3000],
                radius=150,  #orig 150
                #elevation_scale=4,
                #elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                coverage=1
            ),
        ],
    )
)
# CATTLE_DATA = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/nm_cattle.csv"
# POULTRY_DATA = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/nm_chickens.csv"


# HEADER = ["lng", "lat", "weight"]
# cattle_df = pd.read_csv(CATTLE_DATA, header=None).sample(frac=0.5)
# poultry_df = pd.read_csv(POULTRY_DATA, header=None).sample(frac=0.5)

# cattle_df.columns = HEADER
# poultry_df.columns = HEADER

# COLOR_BREWER_BLUE_SCALE = [
#     [240, 249, 232],
#     [204, 235, 197],
#     [168, 221, 181],
#     [123, 204, 196],
#     [67, 162, 202],
#     [8, 104, 172],
# ]


# view = pdk.data_utils.compute_view(cattle_df[["lng", "lat"]])
# view.zoom = 6

# cattle = pdk.Layer(
#     "HeatmapLayer",
#     data=cattle_df,
#     opacity=0.9,
#     get_position=["lng", "lat"],
#     aggregation=pdk.types.String("MEAN"),
#     color_range=COLOR_BREWER_BLUE_SCALE,
#     threshold=1,
#     get_weight="weight",
#     pickable=True,
# )

# poultry = pdk.Layer(
#     "HeatmapLayer",
#     data=poultry_df,
#     opacity=0.9,
#     get_position=["lng", "lat"],
#     threshold=0.75,
#     aggregation=pdk.types.String("MEAN"),
#     get_weight="weight",
#     pickable=True,
# )

# st.pydeck_chart(
#     pdk.Deck(
#     layers=[cattle, poultry],
#     initial_view_state=view,
#     map_provider="mapbox",
#     map_style=pdk.map_styles.SATELLITE,
#     tooltip={"text": "Concentration of cattle in blue, concentration of poultry in orange"},
# )
# )
#r.to_html("heatmap_layer.html")
