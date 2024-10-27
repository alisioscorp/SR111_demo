import os

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

#Centre of geographical domain
lat=41.705881
lon=-3.797353

# SETTING THE ZOOM LOCATIONS 
la_guardia = [41.705881, -3.797353]
jfk = [41.71415, -3.809403]
newark = [41.704989, -3.782342]

#Predefine zoom level for areas
zoom_level = 16

#Blue scales
COLOR_BREWER_BLUE_SCALE = [
    [240, 249, 232],
    [204, 235, 197],
    [168, 221, 181],
    [123, 204, 196],
    [67, 162, 202],
    [8, 104, 172],
]

#Random datasets
df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [lat, lon], columns=["lat", "lon"])  #near Lima
df2 = pd.DataFrame(
    np.random.randn(1000, 2) / [15, 35] + [lat, lon], columns=["lat", "lon"])
df3 = pd.DataFrame(
    np.random.randn(1000, 2) / [30, 20] + [lat, lon], columns=["lat", "lon"])
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
    

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="SR111 Viñedo", page_icon=":wine:")


# LOAD DUMB DATA 
@st.cache_resource
def load_data():
    path = "./uber-raw-data-sep14.csv.gz"
    #if not os.path.isfile(path):
    #    path = f"https://github.com/streamlit/demo-uber-nyc-pickups/raw/main/{path}"

    data = pd.read_csv(
        path,
        nrows=100000,  # approx. 10% of data
        names=[
            "date/time",
            "lat",
            "lon",
        ],  # specify names directly since they don't change
        skiprows=1,  # don't read header since names specified directly
        usecols=[0, 1, 2],  # doesn't load last column, constant value "B02512"
        parse_dates=[
            "date/time"
        ],  # set as datetime instead of converting after the fact
    )
    #data["lon"]=-data["lon"]
    
    return data


# FUNCTION FOR MAPS
def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9", #"mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HeatmapLayer", #"HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    auto_highlight=True,
                    get_radius=50,
                    color_range=COLOR_BREWER_BLUE_SCALE,
                    threshold=.3,
                    pickable=True
                ),
            ],
        )
    )


# FILTER DATA FOR A SPECIFIC HOUR, CACHE
@st.cache_data
def filterdata(df, hour_selected):
    return df[df["date/time"].dt.hour == hour_selected]


# CALCULATE MIDPOINT FOR GIVEN SET OF DATA
@st.cache_data
def mpoint(lat, lon):
    return (np.average(lat), np.average(lon))


# FILTER DATA BY HOUR
@st.cache_data
def histdata(df, hr):
    filtered = data[
        (df["date/time"].dt.hour >= hr) & (df["date/time"].dt.hour < (hr + 1))
    ]

    hist = np.histogram(filtered["date/time"].dt.minute, bins=60, range=(0, 60))[0]

    return pd.DataFrame({"minute": range(60), "pickups": hist})


# STREAMLIT APP LAYOUT
data = load_data()

# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

# SEE IF THERE'S A QUERY PARAM IN THE URL (e.g. ?pickup_hour=2)
# THIS ALLOWS YOU TO PASS A STATEFUL URL TO SOMEONE WITH A SPECIFIC HOUR SELECTED,
# E.G. https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/main?pickup_hour=2
if not st.session_state.get("url_synced", False):
    try:
        pickup_hour = int(st.experimental_get_query_params()["pickup_hour"][0])
        st.session_state["pickup_hour"] = pickup_hour
        st.session_state["url_synced"] = True
    except KeyError:
        pass


# IF THE SLIDER CHANGES, UPDATE THE QUERY PARAM
def update_query_params():
    hour_selected = st.session_state["pickup_hour"]
    st.experimental_set_query_params(pickup_hour=hour_selected)


with row1_1:
    st.title("SR111: Viñedo")
    hour_selected = st.slider(
        "Selecciones día", 0, 14, key="pickup_hour", on_change=update_query_params
    )


with row1_2:
    st.write(
        """
    ##
    Demo de una Recámara de Sibila (SR111) para un viñedo Ribera del Duero.
    Use el control deslizante en la izquierda para visualizar ubicaciones con riesgo de heladas en los últimos 14 días. 
    """
    )

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row2_1, row2_2, row2_3, row2_4 = st.columns((2, 1, 1, 1))

midpoint = mpoint(data["lat"], data["lon"])

with row2_1:
    st.write(
        f"""**Region in Colombia from {hour_selected}:00 and {(hour_selected + 1) % 24}:00**"""
    )
    map(filterdata(data, hour_selected), midpoint[0], midpoint[1], 10)

with row2_2:
    st.write("**Area 1**")
    map(filterdata(data, hour_selected), la_guardia[0], la_guardia[1], zoom_level)

with row2_3:
    st.write("**Area 2**")
    map(filterdata(data, hour_selected), jfk[0], jfk[1], zoom_level)

with row2_4:
    st.write("**Area 3**")
    map(filterdata(data, hour_selected), newark[0], newark[1], zoom_level)

# CALCULATING DATA FOR THE HISTOGRAM
chart_data = histdata(data, hour_selected)



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
            min_zoom=10,
            max_zoom=17,
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
                "HeatmapLayer",#"H3ClusterLayer", #"ScatterplotLayer", #"HeatmapLayer", #'HexagonLayer', #"ScatterplotLayer",
                data=df3,
                #opacity=0.9,
                get_position="[lon, lat]",

                auto_highlight=True,
                get_radius=50,
                color_range=COLOR_BREWER_BLUE_SCALE,
                threshold=.3,
                #get_weight="weight",
                #get_fill_color="[180, 0, 200, 140]",
                #get_color="[180, 0, 200, 140]",
                pickable=True
                #get_color="[200, 30, 0, 160]",
                #elevation_scale=10, 
                ##get_radius=20,
                ##elevation_range=[0, 3000],
                #radius=150,  #orig 150
                ##elevation_scale=4,
                ##elevation_range=[0, 1000],
                #pickable=True,
                #extruded=True,
                #coverage=1
            ),
        ],
    )
,use_container_width=True)

# )
# )
#r.to_html("heatmap_layer.html")
