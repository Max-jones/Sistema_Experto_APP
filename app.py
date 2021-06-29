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

### Initial Confiugurations
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
    layout="wide",
    page_title='Plataforma Abierta - Sistema Experto',
    page_icon='🌊'
    )


# LOADING LOCAL DATA IF EXISTS.
local_path = 'C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\processed/dataframe.csv'



@st.cache(persist=True)

def load_data(path):
    '''
    ARGS: path to the local .csv file
    Load data and search for the Date_Time column to index the dataframe by a datetime value.

    '''
    data = pd.read_csv(path)
    data['Date_Time'] = pd.to_datetime(data['Date_Time'])
    data.set_index('Date_Time', inplace=True)
    chile=pytz.timezone('Chile/Continental')
    data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
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

'''
# Sistema Experto - Plataforma WEB para detección de anomalías
'''

'''
## Cargar el dataset a procesar

'''

import pytz

# Sección de carga del archivo .csv

# Widget para cargar el archivo
uploaded_file = st.file_uploader("Selecciona un archivo .csv ")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])#,format='%Y-%m-%d %H:%m:%S')
    df.set_index('Date_Time', inplace=True)
    chile=pytz.timezone('Chile/Continental')
    df.index = df.index.tz_localize(pytz.utc).tz_convert(chile)
    datas=df
    st.write(
    """
    ##

    **Se ha cargado un archivo. Este debe ser .csv**

    """)
else:
    try:
        datas = load_data(local_path)
    except:

        st.error('Por favor cargue un archivo .csv compatible')
        # raise KeyError('Por favor cargue un archivo .csv compatible')

# Visualización previa del Dataset
'''
## Dataset Seleccionado
'''
with st.beta_expander("Mostrar Dataset Completo"):
    st.write(datas,use_container_width=True)

'''
## Gráficos por variable

'''

p = datas.loc[datas['Etiqueta P'] == 1] #anomaly

import plotly.graph_objects as go

figg = go.Figure()

figg.add_trace(go.Scatter(x=datas.index, y=datas['Pression [cm H2O]'],
                    mode='lines',
                    name='operación normal',
                    line_color='cadetblue'))
figg.add_trace(go.Scatter(x=p.index, y=p['Pression [cm H2O]'],
                    mode='markers',
                    name='anomalía etiquetada',
                    marker_color='cyan',
                    marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg.update_layout(title='Presión [cm H2O]',
                    yaxis_title='Presión [cm H2O]',
                    xaxis_title='Fecha'
)

st.plotly_chart(figg, use_container_width=True)


t = datas.loc[datas['Etiqueta T'] == 1] #anomaly

figg2 = go.Figure()

figg2.add_trace(go.Scatter(x=datas.index, y=datas['Temperatura [°C]'],
                    mode='lines',
                    name='operación normal',
                    line_color='darkolivegreen'))
figg2.add_trace(go.Scatter(x=t.index, y=t['Temperatura [°C]'],
                    mode='markers',
                    name='anomalía etiquetada',
                    marker_color='cyan',
                    marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg2.update_layout(title='Temperatura [°C]',
                    yaxis_title='Temperatura [°C]',
                    xaxis_title='Fecha'
)

st.plotly_chart(figg2, use_container_width=True)

e = datas.loc[datas['Etiqueta EC'] == 1] #anomaly
figg3 = go.Figure()

figg3.add_trace(go.Scatter(x=datas.index, y=datas['EC [µs/cm]'],
                    mode='lines',
                    name='operación normal',
                    line_color='darkgoldenrod'))
figg3.add_trace(go.Scatter(x=e.index, y=e['EC [µs/cm]'],
                    mode='markers',
                    name='anomalía etiquetada',
                    marker_color='cyan',
                    marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg3.update_layout(title='EC [µs/cm]',
                    yaxis_title='EC [µs/cm]',
                    xaxis_title='Fecha'
)


st.plotly_chart(figg3, use_container_width=True)

# row1_1, row1_2, row1_3 = st.beta_columns((2,2,2))

# with row1_1:
#     # p = datas.loc[datas['Etiqueta P'] == 1] #anomaly
#     fig1 = px.line(datas, y='Pression [cm H2O]', title='Presión [cm H2O]')
#     # fig1.add_
#     # px.line()
#     # fig1.add_scatter(p,y='Pression [cm H2O]',title='anomaly')
#     fig1.update_xaxes(
#         rangeslider_visible=True,
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1, label="1m", step="month", stepmode="backward"),
#                 dict(count=6, label="6m", step="month", stepmode="backward"),
#                 dict(count=1, label="YTD", step="year", stepmode="todate"),
#                 dict(count=1, label="1y", step="year", stepmode="backward"),
#                 dict(step="all")
#             ])
#         )
#     )
#     st.plotly_chart(fig1, use_container_width=True) 
# with row1_2:

#     fig2 = px.line(datas, y='Temperatura [°C]',color="Etiqueta T", title='Temperatura [°C]')
#     # px.line()
  
#     fig2.update_xaxes(
#         rangeslider_visible=True,
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1, label="1m", step="month", stepmode="backward"),
#                 dict(count=6, label="6m", step="month", stepmode="backward"),
#                 dict(count=1, label="YTD", step="year", stepmode="todate"),
#                 dict(count=1, label="1y", step="year", stepmode="backward"),
#                 dict(step="all")
#             ])
#         )
#     )
#     st.plotly_chart(fig2, use_container_width=True) 
# with row1_3:
#     fig3 = px.line(datas, y='EC [µs/cm]', title='Conductividad Eléctrica [µs/cm]')
#     # px.line()

#     fig3.update_xaxes(
#         rangeslider_visible=True,
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1, label="1m", step="month", stepmode="backward"),
#                 dict(count=6, label="6m", step="month", stepmode="backward"),
#                 dict(count=1, label="YTD", step="year", stepmode="todate"),
#                 dict(count=1, label="1y", step="year", stepmode="backward"),
#                 dict(step="all")
#             ])
#         )
#     )
#     st.plotly_chart(fig3, use_container_width=True)  



# FILTERING DATA BY HOUR SELECTED
# data = data[data.index.year == year_selected]

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
#  row2_1, row2_2, row2_3, row2_4 = st.beta_columns((2,1,1,1))
with st.beta_expander("Ver análisis estadístico"):
    row2_1, row2_2 = st.beta_columns((2,3))

    # SETTING THE ZOOM LOCATIONS FOR THE LOCATION SITE

    # midpoint

    with row2_1:
        
        '''
        ##

        **Examinando las estadísticas y un mapa(inventado).**

        '''
        zoom_selected = st.slider("Zoom del mapa", 10 , 16)

        st.write('Descripción estadística del dataset cargado.')
        datas_unl=datas.drop(labels=['Etiqueta P','Etiqueta T','Etiqueta EC'],axis=1)
        # datas_raw=datas[["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]]
        st.write(datas_unl.describe())
        # [["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]"]].describe())

        st.write('Datos disponibles',datas_unl.columns.to_list()) 
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(3, 3))
        # sns.pairplot(datas_unl,height=3)
        # st.write(pairplot.fig)

    with row2_2:
        # st.dataframe(datas)
        # horcon= [-32.723230,-71.466365,15]
        map_points = pd.DataFrame(
            np.random.randn(10, 2) / [150, 150] + [-32.723230,-71.466365],
            columns=['lat', 'lon'])
        st.map(map_points,zoom=zoom_selected)
        
        corr = datas_unl.corr()
        heatmap=sns.heatmap(corr, annot=True,cmap="YlGnBu").figure
        st.write(heatmap)

# %% Anomalías
'''
## Detección de anomalías

Se utiliza un modelo pre-entrenado basado en LightGBM sobre toda la data cargada para visualizar anomalías.
'''
loaded_lgbm = lgbm.Booster(model_file='lgb_classifier.txt')

# valores = datas_unl.loc[datas_unl.index[0]].to_numpy().reshape((1,-1))
# st.write(valores[0])
# st.write()
# chart = st.line_chart(valores[0][0])

# for i in range(10): #range(len(datas_unl.index)):

    # input=datas_unl.loc[datas_unl.index[i]].to_numpy().reshape((1,-1))

    # st.pyplot(input)
prob_output=loaded_lgbm.predict(datas_unl.to_numpy())
output = np.int8(prob_output >= 0.5)

new_data = datas_unl.copy()
# st.dataframe(data=new_data)
# new_data =new_data['label']=np.array(output)

b=pd.DataFrame(output,columns=['label'])
# st.write(b)
# st.write(datas_unl)
# st.write(b.columns)
datas_unl['label'] = b.values
new_data.insert(3,'label', b.to_numpy(),True)
# st.write(new_data.columns,new_data.shape)
import matplotlib.pyplot as plt

def read_anomalies(new_data):
    a = new_data.loc[new_data['label'] == 1] #anomaly
    return a

a = read_anomalies(new_data)

fig = plt.figure(figsize=(14,3))
_ = plt.plot(new_data['Pression [cm H2O]'], color='fuchsia', label='Normal')
_ = plt.plot(a['Pression [cm H2O]'], linestyle='none', marker='X', color='orange', markersize=12, label='Anomaly', alpha=0.6)
_ = plt.xlabel('Marca temporal')
_ = plt.ylabel('Sensor Reading')
_ = plt.legend(loc='best')
_ = plt.title('Anomalías sobre Presión ')
st.write(fig)

fig = plt.figure(figsize=(14,3))
_ = plt.plot(new_data['Temperatura [°C]'], color='fuchsia', label='Normal')
_ = plt.plot(a['Temperatura [°C]'], linestyle='none', marker='X', color='orange', markersize=12, label='Anomaly', alpha=0.6)
_ = plt.xlabel('Marca temporal')
_ = plt.ylabel('Sensor Reading')
_ = plt.legend(loc='best')
_ = plt.title('Anomalías sobre Temperatura ')
st.write(fig)

fig = plt.figure(figsize=(14,3))
_ = plt.plot(new_data['EC [µs/cm]'], color='fuchsia', label='Normal')
_ = plt.plot(a['EC [µs/cm]'], linestyle='none', marker='X', color='orange', markersize=12, label='Anomaly', alpha=0.6)
_ = plt.xlabel('Marca temporal')
_ = plt.ylabel('Sensor Reading')
_ = plt.legend(loc='best')
_ = plt.title('Anomalías sobre la conductividad eléctrica ')
st.write(fig)