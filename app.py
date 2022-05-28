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
'''
Creada por Maximiliano Jones
'''



# import pandas_profiling

# Manejo de datos
import pandas as pd
import numpy as np ##
import altair as alt ##
import pydeck as pdk ##

# Funcionalidades de la aplicación
from numpy.lib.function_base import select
import streamlit as st

# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report


# import seaborn as sns


import pytz

# Visualización
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# import plotly.express as px
from bokeh.plotting import figure

# # Clasificadores 
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression

# import lightgbm as lgbm
# import xgboost as xgb

import time

# Model Selection
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import accuracy_score, log_loss

# Automated Classification
from pycaret import classification as supervised
from pycaret import anomaly as unsupervised
# import pycaret.anomaly as unsupervised

    try:
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


    @st.cache
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

    def show_cv_iterations(n_splits, X, y, timeseries=True):
        # https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
        if timeseries:
            cv = TimeSeriesSplit(n_splits)
        else:
            cv = KFold(n_splits)
        
        figure, ax = plt.subplots(figsize=(10, 5))

        for ii, (tr, tt) in enumerate(cv.split(X, y)):
            
            p1 = ax.scatter(tr, [ii] * len(tr), c='black', marker="_", lw=8)
            p2 = ax.scatter(tt, [ii] * len(tt), c='red', marker="_", lw=8)
            ax.set(
                title="Behavior of TimeseriesSplit",
                xlabel="Data Index",
                ylabel="CV Iteration",
                ylim=[5, -1],
            )
            ax.legend([p1, p2], ["Training", "Validation"])
        st.pyplot(fig=figure)
        return cv

    # @st.cache
    def entrenar_modelos(df, etiqueta, metrica, ensamble=True):
        

        # setup
        pycaret_s = supervised.setup(df, target = etiqueta, session_id = 123, silent = True, use_gpu = True, profile = False)     
        # model training and selection
        if ensamble:
            top5 = supervised.compare_models(n_select = 5) 
            # tune top 5 base models
            grid_a= supervised.pull()
            tuned_top5 = [supervised.tune_model(i,fold = 5, optimize='F1',search_library='scikit-optimize') for i in top5]
            grid_b=supervised.pull()
            stacker = supervised.stack_models(estimator_list = top5[1:], meta_model = top5[0])

            # 
            return (stacker, grid_a, grid_b)
        else:
            best = supervised.compare_models(sort= metrica, n_select=3)
            grid = supervised.pull()
            return (best, grid, grid)
        
    def deteccion_no_supervisada(df, metrica, etiqueta=None,  ensamble=True):
        return ""


    def cargar_modelo(df,modelo):

        modelo = supervised.load_model('stack inicial')
        
        return (modelo)

    # Creando las secciones de visualización de la aplicación

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
        labeled = st.sidebar.selectbox(
            "¿El dataset posee etiquetas?",
            ["Seleccione una opción ✔️","Sí", "No"],
            help="Esta pregunta se refiere si la base de datos cargada contiene una columna con la información si los datos han sido etiquetados previamente como datos normales y anómalos.",
        )

        if labeled == "Sí":
            target = st.sidebar.selectbox(
                "Ingrese el nombre de la columna que contiene las etiquetas.",
                columns_names_list,
                help="Esta columna debe ser de tipo binario. Donde 0 corresponde a un dato normal y 1 a una medición anómala.",
                index=len(columns_names_list)-1
            )

        elif labeled == "Seleccione una opción✅":  
            st.sidebar.write("Las preguntas anteriores son obligatorias.")  


        ready = st.sidebar.button("Comenzar!")

        if ready:
    
            selected_df = ds[selected_features]
            if labeled == 'Sí':
                selected_df['target'] = ds[target]
            
            complete_df = selected_df
        
            
            # if st.button("Generar un reporte exploratorio inicial 🕵️"):

                # if st.button('Generar reporte'):
                #     with st.spinner("Training ongoing"):
                #         time.sleep(3)
            st.write('## Análisis exploratorio estadístico y visual de los datos cargados: ')
            with st.expander("🕵️ Mostrar un reporte exploratorio preliminar 📃", expanded=False):
                if st.button("Generar un reporte exploratorio inicial 🕵️"):

            
                # st.write(selected_df)  # use_container_width=True)
                    pr = complete_df.profile_report()
                    # profile = ProfileReport(pr, title="Reporte de exploración de datos")

                    st_profile_report(pr)
                else:
                    st.write('🚧 Por favor seleccione primero las variables a analizar 🚧. ')
            # else:
            #     pass
                

    # %% Separación de los conjuntos de entrenamiento y validación
            
            st.write('## Detección de anomalías')
            if labeled:
                antes = time.time()
                best, grid1, grid2  = entrenar_modelos(complete_df, 'target', 'F1')
                despues = time.time()
                delta_t = despues- antes
                str_t = 'El entrenamiento demoró: '+str(delta_t) + ' segundos.'
                st.write(str_t)
                # pycaret_s = setup(complete_df, target = 'target', session_id = 123, silent = True, use_gpu = True, profile = False)     
                # model training and selection
                # best = compare_models(sort='F1')#,n_select=3)
                # score_grid = pull()
                st.write('### Grilla de búsqueda de modelos:')
                st.write(grid1)
                # st.write(grid2)

                st.write('### Apilamiento de los mejors 5 modelos con mejor desempeño:')
                st.write('# Los mejores clasificador fueron:')
                # st.write(supervised.pull())
                

            # # Guardar modelos
            # save_model(best, 'app_best')
            # st.write(score_grid)
            
            supervised.plot_model(best,plot = 'class_report',display_format='streamlit')
            supervised.plot_model(best,plot = 'confusion_matrix',display_format='streamlit',plot_kwargs = {'percent' : True})
            supervised.plot_model(best,plot = 'error', display_format='streamlit')
            supervised.plot_model(best,plot = 'pr', display_format='streamlit')
            supervised.plot_model(best,plot = 'boundary',display_format='streamlit')
            supervised.plot_model(best,plot = 'calibration',display_format='streamlit')
            # supervised.plot_model(best,plot = 'vc',display_format='streamlit')
            # supervised.plot_model(best,plot = 'feature',display_format='streamlit')
            # supervised.plot_model(best,plot = 'feature_all',display_format='streamlit')
            # supervised.plot_model(best,plot = 'parameter',display_format='streamlit')
            
            
            leaderboard = supervised.get_leaderboard()
            # st.write('Dashboard Resultados:')
            # ds = dashboard(best, display_format='inline')


            # X = selected_df
            # y = ds[target]
            # st.header('Entrenamiento de modelos')
            # # st.write(X)
            # # st.write(y)
            # tscv = show_cv_iterations(5,X,y)



    # %% Comparación de modelos

            # suppervised_classifiers = [
            #     KNeighborsClassifier(3),
            #     SVC(probability=True),
            #     DecisionTreeClassifier(),
            #     RandomForestClassifier(),
            #     AdaBoostClassifier(),
            #     GradientBoostingClassifier(),
            #     GaussianNB(),
            #     LinearDiscriminantAnalysis(),
            #     QuadraticDiscriminantAnalysis(),
            #     LogisticRegression()]

            
            # log_cols = ["Classifier", "Accuracy"]
            # log 	 = pd.DataFrame(columns=log_cols)

            # acc_dict = {}

            # for train_index, test_index in tscv.split(X):
            #     print("TRAIN:", train_index, "TEST:", test_index)
            #     X_train, X_test = X.values[train_index], X.values[test_index]
            #     y_train, y_test = y.values[train_index], y.values[test_index]

            #     for clf in suppervised_classifiers:
            #         # plo
            #         name = clf.__class__.__name__
            #         clf.fit(X_train, y_train)
            #         train_predictions = clf.predict(X_test)
            #         acc = accuracy_score(y_test, train_predictions)
            #         if name in acc_dict:
            #             acc_dict[name] += acc
            #         else:
            #             acc_dict[name] = acc

            # for clf in acc_dict:
            #     acc_dict[clf] = acc_dict[clf] / 10.0
            #     log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
            #     log = log.append(log_entry)
            
            # plt.xlabel('Accuracy')
            # plt.title('Classifier Accuracy')

            # results_fig = plt.figure()
            # sns.set_color_codes("muted")
            # sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
            # st.pyplot(results_fig)
except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")