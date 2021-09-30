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
import requests
import pytz

### Initial Confiugurations
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
    layout="wide",
    page_title='Visualización Datos Acuíferos',
    page_icon='🚰'
    )


# LOADING LOCAL DATA IF EXISTS.
local_path = 'C:\\Users\\elmha\\OneDrive - Universidad de Chile\\Magíster\\Tesis\\Sistema-Experto\\Data\\processed\\dataframe.csv'

# url_queries = 'http://agua.niclabs.cl/queries'

# api_key_header = {'query-api-key': '919c5e5e086a492398141c1ebd95b711'}

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

# def get_info_estacion(estacion):
#     '''
#     ARGS: id estación
#     Returns the Json Info file.
    
#     '''
#     url = url_queries + '/infoestacion'
#     payload = {'estacion':str(estacion)}
#     r = requests.get(url,params=payload,headers=api_key_header)
#     return r.json()

# def get_all_data(estacion):
#     '''
#     ARGS: id estación
#     Returns all de the data into a dataframe.
    
#     '''
#     url = url_queries + '/dataestaciones'
#     payload = {'estacion':str(estacion)}
#     r = requests.get(url,params=payload,headers=api_key_header)
#     return r.json() 

# import json
icon_url='https://www.pngrepo.com/png/276095/180/well.png'
icon_data = {
    # Icon from Wikimedia, used the Creative Commons Attribution-Share Alike 3.0
    # Unported, 2.5 Generic, 2.0 Generic and 1.0 Generic licenses
    "url": icon_url,
    "width": 242,
    "height": 242,
    "anchorY": 242,
}
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
                type="IconLayer",
                data=data,
                get_position=["lon", "lat"],
                get_icon="icon_data",
                get_size=4,
                size_scale=15,
                # elevation_scale=4,
                # elevation_range=[0, 1000],
                pickable=True,
                # extruded=True,
                # colorRange=[[237,248,251],[191,211,230],[158,188,218],[140,150,198],[136,86,167],[129,15,124]]
            )
        ]
    ))

# LAYING OUT THE TOP SECTION OF THE APP

'''
# Visualización Datos Pozo monitoreado Horcón
'''

# '''
# ## Cargar el dataset a procesar


# '''

# with st.beta_expander('Consultar Información de estación en la base de datos'):
#     selected_estacion =st.selectbox(
#         'Seleccione una estación',
#         ('7','1')
#         )

#     r=get_info_estacion(selected_estacion)
#     # _json=get_all_data(selected_estacion)

#     # the json file where the output must be stored 
#     # out_file = open("myfile.json", "w")
#     # json.dump(r2, out_file, indent = 3)  
#     # df_dict=[{'timestamp':item[]}]
#     df = pd.DataFrame(r)
#     st.write(df)




# Sección de carga del archivo .csv

# Widget para cargar el archivo
# uploaded_file = st.file_uploader("Selecciona un archivo .csv ")




df = pd.read_csv(local_path,header=0,engine='python')
# df
df['Date_Time'] = pd.to_datetime(df['Date_Time'])#,format='%Y-%m-%d %H:%m:%S')
df.set_index('Date_Time', inplace=True)
chile=pytz.timezone('Chile/Continental')
df.index = df.index.tz_localize(pytz.utc).tz_convert(chile)
datas=df
# st.write(
# """
# ##

# **Se ha cargado un archivo.**

# """)

row1_1, row1_2 = st.beta_columns((1,2))

    # SETTING THE ZOOM LOCATIONS FOR THE LOCATION SITE

    # midpoint

with row1_1:
    
    '''
    ## **Horcón**
    ### **Id:** 7
    ### **Cliente**: Hidrogeología
    ### **Sector**: Horcón
    ### **Tipo Estación**: Pozo
    ### **Latitud**: -32.70846697
    ### **Longitud**: -71.49001948

    '''
    

    # st.write('Descripción estadística del dataset cargado.')
    # datas_unl=datas.drop(labels=['Etiqueta P','Etiqueta T','Etiqueta EC'],axis=1)
    # # datas_raw=datas[["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]]
    # st.write(datas_unl.describe())
    # # [["Pression [cm H2O]","Temperatura [°C]","EC [µs/cm]"]].describe())

    # st.write('Datos disponibles',datas_unl.columns.to_list()) 
with row1_2:
    # st.dataframe(datas)
    horcon= [[-32.70846697,-71.49001948]]
    map_points = pd.DataFrame(
        horcon,
        columns=['lat', 'lon'])
    # st.map(map_points,13,)
    map(map_points,horcon[0][0],horcon[0][1],18)
        
# '''
# ## Dataset Seleccionado
# '''
# with st.beta_expander("Mostrar Dataset Completo"):
#     st.write(datas,use_container_width=True)

'''
## Gráficos por variable

'''

p = datas.loc[datas['Etiqueta P'] == 1] #anomaly



figg = go.Figure()

figg.add_trace(go.Scatter(x=datas.index, y=datas['Pression [cm H2O]'],
                    mode='lines',
                    name='operación normal',
                    line_color='cadetblue'))
# figg.add_trace(go.Scatter(x=p.index, y=p['Pression [cm H2O]'],
#                     mode='markers',
#                     name='anomalía etiquetada',
#                     marker_color='cyan',
#                     marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg.update_layout(title='Presión [cm H2O]',
                    yaxis_title='Presión [cm H2O]',
                    xaxis_title='Fecha'
)
figg.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="todate"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(figg, use_container_width=True)


t = datas.loc[datas['Etiqueta T'] == 1] #anomaly

figg2 = go.Figure()

figg2.add_trace(go.Scatter(x=datas.index, y=datas['Temperatura [°C]'],
                    mode='lines',
                    name='operación normal',
                    line_color='darkolivegreen'))
# figg2.add_trace(go.Scatter(x=t.index, y=t['Temperatura [°C]'],
#                     mode='markers',
#                     name='anomalía etiquetada',
#                     marker_color='cyan',
#                     marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg2.update_layout(title='Temperatura [°C]',
                    yaxis_title='Temperatura [°C]',
                    xaxis_title='Fecha'
)
figg2.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="todate"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(figg2, use_container_width=True)


e = datas.loc[datas['Etiqueta EC'] == 1] #anomaly

figg3 = go.Figure()

figg3.add_trace(go.Scatter(x=datas.index, y=datas['EC [µs/cm]'],
                    mode='lines',
                    name='operación normal',
                    line_color='darkgoldenrod'))
# figg3.add_trace(go.Scatter(x=e.index, y=e['EC [µs/cm]'],
#                     mode='markers',
#                     name='anomalía etiquetada',
#                     marker_color='cyan',
#                     marker_line_width=0.5))
# figg.update_traces(mode='markers', marker_line_width=2, marker_size=10)
figg3.update_layout(title='Conductividad Eléctrica [µs/cm]',
                    yaxis_title='CE [µs/cm]',
                    xaxis_title='Fecha'
)
figg3.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="todate"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(figg3, use_container_width=True)


with st.beta_expander("Ver análisis estadístico"):
    
    '''
    ##**Examinando algunas estadísticas.**
    '''
    row2_1, row2_2, row2_3 = st.beta_columns((2,2,2))


    with row2_1:
        hist_p = px.histogram(datas, x='Pression [cm H2O]',nbins=20)
        st.plotly_chart(hist_p)
  
        datas_unl=datas.drop(labels=['Etiqueta P','Etiqueta T','Etiqueta EC'],axis=1)

        st.write('Datos disponibles',datas_unl.columns.to_list())

        st.write('Descripción estadística del dataset cargado.')

        st.write(datas_unl.describe())

 

    with row2_2:
        hist_t = px.histogram(datas, x='Temperatura [°C]',nbins=20)
        st.plotly_chart(hist_t)
        # st.dataframe(datas)
        # horcon= [-32.723230,-71.466365,15]
        # map_points = pd.DataFrame(
        #     np.random.randn(10, 2) / [150, 150] + [-32.723230,-71.466365],
        #     columns=['lat', 'lon'])
        # st.map(map_points,zoom=zoom_selected)
        
        corr = datas_unl.corr()
        conf_mat = px.imshow(corr)
        st.plotly_chart(conf_mat)
        st.write(corr)



        # heatmap=sns.heatmap(corr, annot=True,cmap="YlGnBu").figure
        # st.write(heatmap)
    with row2_3:
        hist_ec = px.histogram(datas, x='EC [µs/cm]',nbins=20)
        st.plotly_chart(hist_ec)
        

with st.beta_expander("Sub sampling"):#,expanded=True):
    # Plotear un subsampling de 3 h
    '''
    ## Realizando un subsampling de 3 horas, 12 horas y 1 día
    '''
    # 3 horas
    df_ss_3h_raw = df.resample(rule='3H',convention='end',closed='right')

    roll_3h_mean = df_ss_3h_raw.mean()
    roll_3h_std = df_ss_3h_raw.std()

    df_ss_3h = df_ss_3h_raw.asfreq()


    # 12 horas
    df_ss_12h_raw = df.resample(rule='12H',convention='end',closed='right')

    roll_12h_mean = df_ss_12h_raw.mean()
    roll_12h_std = df_ss_12h_raw.std()

    df_ss_12h = df_ss_12h_raw.asfreq()

    # diario
    df_ss_1d_raw = df.resample(rule='D')
    roll_1d_mean = df_ss_1d_raw.mean()
    roll_1d_std = df_ss_1d_raw.std()

    df_ss_1d = df_ss_1d_raw.asfreq()


    figg4 = go.Figure()
    figg4.add_trace(go.Scatter(x=datas.index, y=datas['Pression [cm H2O]'],
                        mode='lines',
                        name='operación normal'))

    figg4.add_trace(go.Scatter(x=df_ss_3h.index, y=df_ss_3h['Pression [cm H2O]'],
                        mode='lines',
                        name='subsampling 3 horas',
                        line_color='darkolivegreen'))
    figg4.add_trace(go.Scatter(x=df_ss_12h.index, y=df_ss_12h['Pression [cm H2O]'],
                    mode='lines',
                    name='subsampling 12 horas',
                    line_color='darkgoldenrod'))
    figg4.add_trace(go.Scatter(x=df_ss_1d.index, y=df_ss_1d['Pression [cm H2O]'],
                mode='lines',
                name='subsampling 1 día',
                line_color='cadetblue'))


    figg4.update_layout(title='Presión [cm H2O]',
                    yaxis_title='Presión [cm H2O]',
                    xaxis_title='Fecha'
    )
    figg4.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="todate"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(figg4, use_container_width=True)
    # st.write(rollmean,use_container_width=True)
    # st.write(rollstd,use_container_width=True)

    figg5 = go.Figure()
    figg5.add_trace(go.Scatter(x=datas.index, y=datas['EC [µs/cm]'],
                        mode='lines',
                        name='operación normal'))

    figg5.add_trace(go.Scatter(x=df_ss_3h.index, y=df_ss_3h['EC [µs/cm]'],
                        mode='lines',
                        name='subsampling 3 horas',
                        line_color='darkolivegreen'))
    figg5.add_trace(go.Scatter(x=df_ss_12h.index, y=df_ss_12h['EC [µs/cm]'],
                    mode='lines',
                    name='subsampling 12 horas',
                    line_color='darkgoldenrod'))
    figg5.add_trace(go.Scatter(x=df_ss_1d.index, y=df_ss_1d['EC [µs/cm]'],
                mode='lines',
                name='subsampling 1 día',
                line_color='cadetblue'))


    figg5.update_layout(title='CE [µs/cm]',
                    yaxis_title='CE [µs/cm]',
                    xaxis_title='Fecha'
    )
    figg5.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="todate"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(figg5, use_container_width=True)

    figg6 = go.Figure()
    figg6.add_trace(go.Scatter(x=datas.index, y=datas['Temperatura [°C]'],
                        mode='lines',
                        name='operación normal'))

    figg6.add_trace(go.Scatter(x=df_ss_3h.index, y=df_ss_3h['Temperatura [°C]'],
                        mode='lines',
                        name='subsampling 3 horas',
                        line_color='darkolivegreen'))
    figg6.add_trace(go.Scatter(x=df_ss_12h.index, y=df_ss_12h['Temperatura [°C]'],
                    mode='lines',
                    name='subsampling 12 horas',
                    line_color='darkgoldenrod'))
    figg6.add_trace(go.Scatter(x=df_ss_1d.index, y=df_ss_1d['Temperatura [°C]'],
                mode='lines',
                name='subsampling 1 día',
                line_color='cadetblue'))


    figg6.update_layout(title='Temperatura [°C]',
                    yaxis_title='Temperatura [°C]',
                    xaxis_title='Fecha'
    )
    figg6.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="todate"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(figg6, use_container_width=True)

# %% Anomalías
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

