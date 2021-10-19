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
import pandas_profiling

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
import requests
import pytz
import time
import random

from streamlit_pandas_profiling import st_profile_report

### Initial Confiugurations
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
    layout="wide",
    page_title='Plataforma automática para detección de anomalías',
    page_icon='🚀',
    initial_sidebar_state="expanded",
    )

# LOADING LOCAL DATA IF EXISTS.
local_path = 'C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\processed/dataframe.csv'


# @st.cache
def load_data(path):
    '''
    ARGS: path to the local .csv file
    Load data and search for the Date_Time column to index the dataframe by a datetime value.

    '''
    data = pd.read_csv(path,sep=",")#, engine='python')
    data['Date_Time'] = pd.to_datetime(data['Date_Time'])
    data.set_index('Date_Time', inplace=True)
    chile=pytz.timezone('Chile/Continental')
    data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
    return data

# CREATING FUNCTION FOR MAPS


def map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style='mapbox://styles/mapbox/outdoors-v11',
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50
        },
        tooltip={"text": "Horcón {}, {}\n Mediciones disponibles: \n CE, Temp, Nivel".format(lat,lon)},
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
                colorRange=[[237,248,251],[191,211,230],[158,188,218],[140,150,198],[136,86,167],[129,15,124]]
            )
        ]
    ))
# with st.sidebar():
    # self.sidebar.radio('El d',[1,2,3,4])
# LAYING OUT THE TOP SECTION OF THE APP


# Título de la plataforma
'''
# Sistema Experto - Plataforma WEB para detección de anomalías
'''

st.sidebar.write('# Menu de pre-configuración')
st.sidebar.write(
'''
## Cargar el dataset a procesar
'''
)

# Sección de carga del archivo .csv

    # Widget para cargar el archivo
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo .csv ")

if uploaded_file is not None:
    uploaded_file.seek(0)
    ds=load_data(uploaded_file)

    st.sidebar.write("**Se ha cargado un archivo.**")
    columns_names_list = ds.columns.to_list()
    st.sidebar.write(columns_names_list)
    # st.sidebar.write(type(columns_names_list))
    selected_features = st.sidebar.multiselect('Seleccione las columnas que contienen nombres de características', columns_names_list)


    # '''
    # ## Dataset Seleccionado
    # '''
    if st.sidebar.button('Seleccionar columnas'):

        # if st.button('Generar reporte'):
        #     with st.spinner("Training ongoing"):       
        #         time.sleep(3)
        with st.beta_expander("Mostrar Dataset Completo",expanded=True):
                    selected_df = ds[selected_features]
                    st.write(selected_df)#use_container_width=True)
                    pr = selected_df.profile_report()

                    st_profile_report(pr)
    # '''
    # ## Gráficos por variable

    # '''

    # p = datas.loc[datas['Etiqueta P'] == 1] #anomaly

    # figg = go.Figure()

    # figg.add_trace(go.Scatter(x=datas.index, y=datas['Pression [cm H2O]'],
    #                     mode='lines',
    #                     name='operación normal',
    #                     line_color='cadetblue'))
    # # figg.add_trace(go.Scatter(x=p.index, y=p['Pression [cm H2O]'],
    # #                     mode='markers',
    # #                     name='anomalía etiquetada',
    # #                     marker_color='cyan',
    # #                     marker_line_width=0.5))
    # # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    # figg.update_layout(title='Presión [cm H2O]',
    #                     yaxis_title='Presión [cm H2O]',
    #                     xaxis_title='Fecha'
    # )

    # st.plotly_chart(figg, use_container_width=True)


    # t = datas.loc[datas['Etiqueta T'] == 1] #anomaly

    # figg2 = go.Figure()

    # figg2.add_trace(go.Scatter(x=datas.index, y=datas['Temperatura [°C]'],
    #                     mode='lines',
    #                     name='operación normal',
    #                     line_color='darkolivegreen'))
    # # figg2.add_trace(go.Scatter(x=t.index, y=t['Temperatura [°C]'],
    # #                     mode='markers',
    # #                     name='anomalía etiquetada',
    # #                     marker_color='cyan',
    # #                     marker_line_width=0.5))
    # # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    # figg2.update_layout(title='Temperatura [°C]',
    #                     yaxis_title='Temperatura [°C]',
    #                     xaxis_title='Fecha'
    # )

    # st.plotly_chart(figg2, use_container_width=True)


    # e = datas.loc[datas['Etiqueta EC'] == 1] #anomaly

    # figg3 = go.Figure()

    # figg3.add_trace(go.Scatter(x=datas.index, y=datas['EC [µs/cm]'],
    #                     mode='lines',
    #                     name='operación normal',
    #                     line_color='darkgoldenrod'))
    # # figg3.add_trace(go.Scatter(x=e.index, y=e['EC [µs/cm]'],
    # #                     mode='markers',
    # #                     name='anomalía etiquetada',
    # #                     marker_color='cyan',
    # #                     marker_line_width=0.5))
    # # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    # figg3.update_layout(title='EC [µs/cm]',
    #                     yaxis_title='EC [µs/cm]',
    #                     xaxis_title='Fecha'
    # )

    # st.plotly_chart(figg3, use_container_width=True)


    # with st.beta_expander("Ver análisis estadístico"):
    #     row2_1, row2_2 = st.beta_columns((2,3))

    #     # SETTING THE ZOOM LOCATIONS FOR THE LOCATION SITE

    #     # midpoint

    #     with row2_1:
            
    #         '''
    #         ##

    #         **Examinando las estadísticas y un mapa(inventado).**

    #         '''
    #         zoom_selected = st.slider("Zoom del mapa", 10 , 16)

    #         st.write('Descripción estadística del dataset cargado.')
    #         datas_unl=datas.drop(labels=['Etiqueta P','Etiqueta T','Etiqueta EC'],axis=1)
    #         # datas_raw=datas[["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]]
    #         st.write(datas_unl.describe())
    #         # [["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]"]].describe())

    #         st.write('Datos disponibles',datas_unl.columns.to_list()) 
    #         # import matplotlib.pyplot as plt
    #         # plt.figure(figsize=(3, 3))
    #         # sns.pairplot(datas_unl,height=3)
    #         # st.write(pairplot.fig)

    #     with row2_2:
    #         # st.dataframe(datas)
    #         # horcon= [-32.723230,-71.466365,15]
    #         # map_points = pd.DataFrame(
    #         #     np.random.randn(10, 2) / [150, 150] + [-32.723230,-71.466365],
    #         #     columns=['lat', 'lon'])
    #         # st.map(map_points,zoom=zoom_selected)
            
    #         corr = datas_unl.corr()
    #         heatmap=sns.heatmap(corr, annot=True,cmap="YlGnBu").figure
    #         st.write(heatmap)

    # # %% Anomalías
    # with st.beta_expander("Procesar Anomalías",expanded=True):

    #     '''
    #     ## Detección de anomalías

    #     Se utiliza un modelo pre-entrenado basado en LightGBM sobre toda la data cargada para detectar y visualizar anomalías.
    #     '''
    #     loaded_lgbm = lgbm.Booster(model_file='lgb_classifier.txt')

    #     prob_output=loaded_lgbm.predict(datas_unl.to_numpy())
    #     output = np.int8(prob_output >= 0.5)

    #     new_data = datas_unl.copy()
    #     # st.dataframe(data=new_data)
    #     # new_data =new_data['label']=np.array(output)

    #     b=pd.DataFrame(output,columns=['label'])
    #     # st.write(b)
    #     # st.write(datas_unl)
    #     # st.write(b.columns)
    #     datas_unl['etiqueta_anomalía'] = b.values
    #     new_data.insert(3,'etiqueta_anomalia', b.to_numpy(),True)
    #     # st.write(new_data.columns,new_data.shape)
    #     import matplotlib.pyplot as plt

    #     def read_anomalies(new_data):
    #         a = new_data.loc[new_data['etiqueta_anomalia'] == 1] #anomaly
    #         return a

    #     a = read_anomalies(new_data)

    #     st.write(new_data)

    #     p = datas.loc[datas['Etiqueta P'] == 1] #anomaly

    #     import plotly.graph_objects as go

    #     figg = go.Figure()

    #     figg.add_trace(go.Scatter(x=datas.index, y=datas['Pression [cm H2O]'],
    #                         mode='lines',
    #                         name='operación normal',
    #                         line_color='cadetblue'))
    #     figg.add_trace(go.Scatter(x=p.index, y=p['Pression [cm H2O]'],
    #                         mode='markers',
    #                         name='anomalía etiquetada',
    #                         marker_color='cyan',
    #                         marker_line_width=0.5,
    #                         opacity=0.5))
    #     figg.add_trace(go.Scatter(x=a.index, y=a['Pression [cm H2O]'],
    #                         mode='markers',
    #                         name='anomalía detectada',
    #                         marker_color='red',
    #                         marker_line_width=0.5,
    #                         opacity=0.7))
                                                
    #     # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    #     figg.update_layout(title='Presión [cm H2O]',
    #                         yaxis_title='Presión [cm H2O]',
    #                         xaxis_title='Fecha'
    #     )

    #     st.plotly_chart(figg, use_container_width=True)


    #     t = datas.loc[datas['Etiqueta T'] == 1] #anomaly

    #     figg2 = go.Figure()

    #     figg2.add_trace(go.Scatter(x=datas.index, y=datas['Temperatura [°C]'],
    #                         mode='lines',
    #                         name='operación normal',
    #                         line_color='darkolivegreen'))
    #     figg2.add_trace(go.Scatter(x=t.index, y=t['Temperatura [°C]'],
    #                         mode='markers',
    #                         name='anomalía etiquetada',
    #                         marker_color='cyan',
    #                         marker_line_width=0.5,
    #                         opacity=0.5))
    #     figg2.add_trace(go.Scatter(x=a.index, y=a['Temperatura [°C]'],
    #                         mode='markers',
    #                         name='anomalía detectada',
    #                         marker_color='red',
    #                         marker_line_width=0.5,
    #                         opacity=0.7))        
    #     # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    #     figg2.update_layout(title='Temperatura [°C]',
    #                         yaxis_title='Temperatura [°C]',
    #                         xaxis_title='Fecha'
    #     )

    #     st.plotly_chart(figg2, use_container_width=True)

    #     e = datas.loc[datas['Etiqueta EC'] == 1] #anomaly
    #     figg3 = go.Figure()

    #     figg3.add_trace(go.Scatter(x=datas.index, y=datas['EC [µs/cm]'],
    #                         mode='lines',
    #                         name='operación normal',
    #                         line_color='darkgoldenrod'))
    #     figg3.add_trace(go.Scatter(x=e.index, y=e['EC [µs/cm]'],
    #                         mode='markers',
    #                         name='anomalía etiquetada',
    #                         marker_color='cyan',
    #                         marker_line_width=0.5,
    #                         opacity=0.5))
    #     figg3.add_trace(go.Scatter(x=a.index, y=a['EC [µs/cm]'],
    #                         mode='markers',
    #                         name='anomalía detectada',
    #                         marker_color='red',
    #                         marker_line_width=0.5,
    #                         opacity=0.7))                     
    #     # figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    #     figg3.update_layout(title='EC [µs/cm]',
    #                         yaxis_title='EC [µs/cm]',
    #                         xaxis_title='Fecha'
    #     )


    #     st.plotly_chart(figg3, use_container_width=True)

