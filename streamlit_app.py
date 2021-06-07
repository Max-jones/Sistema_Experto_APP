# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)
path = 'C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\Raw\Horcon-etiquetado_v2.xlsx'

@st.cache(persist=True)

def load_data():
    data = pd.read_excel(path,header=3)# parse_dates=[['Date', 'Time']], usecols=range(8))
#     # lowercase = lambda x: str(x).lower()
#     # data.rename(lowercase, axis="columns", inplace=True)
#     data.set_index('Date_Time',inplace=True)
#     return data
    return data

data = []
datas = load_data()
# CREATING FUNCTION FOR MAPS

def map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.beta_columns((2,3))

with row1_1:
    st.title("Sistema Experto - Visualizaciones - Pruebas")
    zoom_selected = st.slider("Zoom", 10 , 20)
    year_selected = st.slider("Select year", 2014, 2017)

with row1_2:
    st.write(
    """
    ##

    **Examinando las visualizaciones y un mapa**

    """)
    st.dataframe(datas)


# FILTERING DATA BY HOUR SELECTED
# data = data[data.index.year == year_selected]

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
#  row2_1, row2_2, row2_3, row2_4 = st.beta_columns((2,1,1,1))
row2_1, row2_2, = st.beta_columns((2,3))

# SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
horcon= [-32.723230,-71.466365,15]
# midpoint

with row2_1:
#     st.write("**All New York City from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
#     map(data, midpoint[0], midpoint[1], 11)
    st.write("**Horcón**")
    map(data, horcon[0],horcon[1], zoom_selected)

with row2_2:
    st.dataframe(datas)



# FILTERING DATA FOR THE HISTOGRAM
# filtered = data[
#     (data.index.month >= month_selected) & (data.index.month < (month_selected + 1))
#     ]

# hist = np.histogram(filtered.index.day, bins=25, range=(0, 30))[0]

# chart_data = pd.DataFrame({"minute": range(60), "Some Value": hist})

# # LAYING OUT THE HISTOGRAM SECTION

# st.write("")

# st.write("**Breakdown of rides per minute between %i:00 and %i:00**" % (year_selected, (month_selected + 1) % 24))

# st.altair_chart(alt.Chart(chart_data)
#     .mark_area(
#         interpolate='step-after',
#     ).encode(
#         x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
#         y=alt.Y("pickups:Q"),
#         tooltip=['minute', 'pickups']
#     ).configure_mark(
#         opacity=0.5,
#         color='red'
#     ), use_container_width=True)

