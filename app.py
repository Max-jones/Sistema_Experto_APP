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
from bokeh.plotting import figure
import plotly.figure_factory as ff
import plotly.express as px

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
path = 'C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\processed/dataframe.csv'
# year_selected=2015

@st.cache(persist=True)

def load_data():
    data = pd.read_csv(path)
    data['Date_Time'] = pd.to_datetime(data['Date_Time'])
    data.set_index('Date_Time', inplace=True)
    # data[str(year_selected)]
    return data


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

st.title("Sistema Experto - Visualizaciones - Pruebas")

zoom_selected = st.slider("Zoom del mapa", 10 , 16)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df.set_index('Date_Time', inplace=True)
    datas=df
    st.write(
    """
    ##

    **Se ha cargado un archivo. Este debe ser .csv**

    """)
else:
    try:
        datas = load_data()
    except:

        st.error('Por favor cargue un archivo .csv compatible')
        # raise KeyError('Por favor cargue un archivo .csv compatible')


row1_1, row1_2 = st.beta_columns((2,3))

with row1_1:
    # st.title("Sistema Experto - Visualizaciones - Pruebas")
    # zoom_selected = st.slider("Zoom", 10 , 20)
    # # year_selected = st.slider("Select year", 2014, 2017)
    # year_selected = 2014
    fig2 = px.line(datas, y='Temperatura [°C]', title='Temperatura')
    # px.line()

    fig2.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]))
    )
    st.plotly_chart(fig2, use_container_width=True) 
with row1_2:
    st.write(
    """
    ##

    **Examinando las visualizaciones y un mapa**

    """)
    st.dataframe(data=datas.describe())




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
    # st.write("**Horcón**")
    # map(data, horcon[0],horcon[1], zoom_selected)

    # x = [1, 2, 3, 4, 5]
    # y = [6, 7, 2, 4, 5]
    # p = figure(
    #     title='simple line example',
    #     x_axis_label='x',
    #     y_axis_label='y')
    # p.line(x, y, legend_label='Trend', line_width=2)
    # st.bokeh_chart(p, use_container_width=True)
    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2
    
    # Group data together
    hist_data = [x1]
    
    group_labels = ['Group 1', 'Group 2', 'Group 3']
    
    # Create distplot with custom bin_size
    # fig = ff.create_distplot(
    #     # hist_data, group_labels, bin_size=[.1, .25, .5])
    fig = px.line(datas, y='Pression [cm H2O]', title='Presión')
    # px.line()

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]))
    )
    
    # fig.show()    
    
    # Plot!
    st.plotly_chart(fig, use_container_width=True)    
     

with row2_2:
    # st.dataframe(datas)
    df = pd.DataFrame(
        np.random.randn(10, 2) / [150, 150] + [-32.723230,-71.466365],
        columns=['lat', 'lon'])
    st.map(df,zoom=zoom_selected)


