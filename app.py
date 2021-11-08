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

# %% Imports -> requeriments.txt

import streamlit as st
import pandas as pd

import numpy as np
import altair as alt
import pydeck as pdk
from bokeh.plotting import figure
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import lightgbm as lgbm
from streamlit.proto.DataFrame_pb2 import DataFrame
import plotly.graph_objects as go
import pytz


from streamlit_pandas_profiling import st_profile_report

### Initial Confiugurations
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
    layout="wide",
    page_title="Plataforma automática para detección de anomalías",
    page_icon="🚀",
    initial_sidebar_state="expanded",
)

# LOADING LOCAL DATA IF EXISTS.
# local_path = "C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\processed/dataframe.csv"


# @st.cache
def load_data(path):
    """
    ARGS: path to the local .csv file
    Load data and search for the Date_Time column to index the dataframe by a datetime value.

    """
    data = pd.read_csv(path,delimiter=";")  # , engine='python')
    data["Date_Time"] = pd.to_datetime(data["Date_Time"])
    data.set_index("Date_Time", inplace=True)
    chile = pytz.timezone("Chile/Continental")
    data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
    return data


# CREATING FUNCTION FOR MAPS


def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/outdoors-v11",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            tooltip={
                "text": "Horcón {}, {}\n Mediciones disponibles: \n CE, Temp, Nivel".format(
                    lat, lon
                )
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    radius=20,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                    colorRange=[
                        [237, 248, 251],
                        [191, 211, 230],
                        [158, 188, 218],
                        [140, 150, 198],
                        [136, 86, 167],
                        [129, 15, 124],
                    ],
                )
            ],
        )
    )


# LAYING OUT THE TOP SECTION OF THE APP


# Título de la plataforma
"""
# Sistema Experto - Plataforma WEB para detección de anomalías
"""

st.sidebar.write("## Menu de pre-configuración")
st.sidebar.write(
    """
### 1️⃣ Cargar el dataset a procesar
"""
)

# Sección de carga del archivo .csv

# Widget para cargar el archivo
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo .csv ")

# La aplicación comienza cuando se carga un archivo.
if uploaded_file is not None:
    uploaded_file.seek(0)

    # Se carga el archivo
    ds = load_data(uploaded_file)

    # Confirmación carga archivo
    st.sidebar.write("**Se ha cargado un archivo.**")

    # Se extraen los nombres de las columnas del dataset cargado.
    columns_names_list = ds.columns.to_list()
    st.sidebar.write(columns_names_list)

    # Widget para seleccionar las variables monitoreadas a analizar.
    st.sidebar.write(
    """
    ### 2️⃣ Seleccione los nombres de las columnas que contienen características
    """)

    selected_features = st.sidebar.multiselect(
        " Seleccione las características",
        columns_names_list,
    )
    
    # Widget de consulta si el dataset contiene etiquetas.
    supervised = st.sidebar.selectbox(
        "¿El dataset posee etiquetas?",
        ["Seleccione una opción✅","Sí", "No"],
        help="Esta pregunta se refiere si la base de datos cargada contiene una columna con la información si los datos han sido etiquetados previamente como datos normales y anómalos.",
    )

    if supervised == "Sí":
        target = st.sidebar.selectbox(
            "Ingrese el nombre de la columna que contiene las etiquetas.",
            columns_names_list,
            help="Esta columna debe ser de tipo binario. Donde 0 corresponde a un dato normal y 1 a una medición anómala.",
        )

    elif supervised == "Seleccione una opción✅":  
        st.sidebar.write("Las preguntas anteriores son obligatorias.")  


    # ready = st.sidebar.button("Comenzar!")

    # if ready:

        # if selected_features != []:
    if st.button("Mostrar un reporte exploratorio inicial"):

        # if st.button('Generar reporte'):
        #     with st.spinner("Training ongoing"):
        #         time.sleep(3)
        # with st.beta_expander("🕵️ Mostrar un reporte exploratorio inicial 📃", expanded=True):
        selected_df = ds[selected_features]
        st.write(selected_df)  # use_container_width=True)
        pr = selected_df.profile_report()

        st_profile_report(pr)
        # else:
        #     st.write('🚧 Por favor seleccione primero las variables a analizar 🚧. ')
    