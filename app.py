# -*- coding: utf-8 -*-

# Creada por Maximiliano Jones

# Manejo de datos
from ast import parse
import pandas as pd

# Funcionalidades de la aplicación
import streamlit as st
import base64
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Manejod del tiempo/fechas
import pytz
import time


# Automated Classification
from pycaret import classification as supervised
# import pycaret.anomaly as unsupervised


import plotly.express as px

# Funciones auxiliares

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


@st.cache(suppress_st_warning=True)
def load_data(path):
    '''
    ARGS: path to the local .csv file
    Load data and search for the Date_Time column to index the dataframe by a datetime value.

    '''

    data = pd.read_csv(path, sep=None, engine='python',encoding = 'utf-8-sig',parse_dates= True)

    try:
        data['Date_Time'] = pd.to_datetime(data['Date_Time'],dayfirst=True)
        st.sidebar.write('Se encontró una columa "Date_time"')
        data.set_index("Date_Time", inplace=True)
        chile = pytz.timezone("Chile/Continental")
        data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
        st.dataframe(data)
        return data
    except:
        try:
            data['Datetime'] = pd.to_datetime(data["Date_Time"],)
            st.sidebar.write('Se encontró una columa "Datetime"')
            data.set_index("Datetime", inplace=True)
            chile = pytz.timezone("Chile/Continental")
            data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
            st.dataframe(data)
            return data
        except:
            st.write("Se entró en el tercer except")
            st.sidebar.write("No se encontró columna Date_Time")
            return data

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def entrenar_modelos(df, etiqueta, metrica, ensamble=True, debug=True):

    '''
    ARGS: dataframe (pd.DataFrame),
    etiqueta con nombre de dataframe.column (str),
    metrica puede ser ['f1', 'accuracy', 'recall'] (str) y
    ensamble[default=True, False] (boolean)
    '''

    # setup
    pycaret_s = supervised.setup(df, target=etiqueta, session_id=123, silent=True, use_gpu=False, profile=False, log_experiment=False, fix_imbalance=True)
    # model training and selection
    if ensamble:
        # with st.snow():
        top10 = supervised.compare_models(n_select=10)
        top5 = top10[0:4]
        # tune top 5 base models
        grid_a = supervised.pull()
        tuned_top5 = [supervised.tune_model(i, fold=5, optimize='F1', search_library='scikit-optimize') for i in top5]
        # grid_b = supervised.pull()
        stacker = supervised.stack_models(estimator_list=tuned_top5[1:], meta_model=tuned_top5[0])
        if debug:
            st.write(top10)
            # st.write(grid_b)
        # else:
        #     pass
            
        #
        return (stacker, grid_a, grid_a)
    else:
        best = supervised.compare_models(sort=metrica, n_select=3)
        grid = supervised.pull()
        return (best, grid, grid)


# def deteccion_no_supervisada(df, metrica, etiqueta=None, ensamble=True):
#     return ""

# def cargar_modelo(df, modelo):
    modelo = supervised.load_model('stack inicial')

    return (modelo)


colors_blue = ["#132C33", "#264D58", '#17869E', '#51C4D3', '#B4DBE9']
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

def generar_distrib(df, etiqueta):
    figura = px.histogram(df,x=etiqueta,y=df[etiqueta],color='Etiqueta',template='plotly_white',
                    marginal='box',opacity=0.7,nbins=100,color_discrete_sequence=[colors_green[3],colors_blue[3]],
                    barmode='group',histfunc='count')
                    
    figura.update_layout(
        font_family='monospace',
        title=dict(text=etiqueta,x=0.53,y=0.95,
                font=dict(color=colors_dark[2],size=20)),
        xaxis_title_text=etiqueta,
        yaxis_title_text='Count',
        legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
        bargap=0.3,
    )
    return figura

try:
    ### Initial Confiugurations
    # SETTING PAGE CONFIG TO WIDE MODE
    st.set_page_config(
        layout="wide",
        page_title="Plataforma automática para detección de anomalías",
        page_icon="🚀",
        initial_sidebar_state="expanded",
    )
    set_bg_hack('images/app_background.png')
    # LOADING LOCAL DATA IF EXISTS.
    # local_path = "C:\\Users\elmha\OneDrive - Universidad de Chile\Magíster\Tesis\Sistema-Experto\Data\processed/dataframe.csv"

    # Creando las secciones de visualización de la aplicación
    # %%%
    # Título de la plataforma
    '''
    # Sistema Experto - Plataforma WEB para detección de anomalías
    '''

    st.image('images/metodologia3.png')
    info = st.info("""Instrucciones:
          
          1. El primer paso es cargar el set de datos a procesar en el apartado 1️⃣ del panel lateral. 
          NOTA: El archivo debe estar en formato .csv y no debe ter filas vacías antes de las columnas de cabecera de los datos.
    2. Ahora se deben seleccionar los nombres de las columnas o atributos con información relevante para la detección de anomalías en el aparato 2️⃣.
    3. Finalmente se debe seleccionar la etiqueta o el target de las anomalías etiquetadas y hacer click en "Comenzar". """)

    st.sidebar.write("## Configuración inicial")
    st.sidebar.write(
        """
    ### 1️⃣ Cargar el dataset a procesar
    """
    )

    # Sección de carga del archivo .csv

    # Widget para cargar el archivo
    uploaded_file = st.sidebar.file_uploader("Selecciona un archivo .csv ", type='csv')

    # La aplicación comienza cuando se carga un archivo.

    if uploaded_file is not None:

        uploaded_file.seek(0)

        # Se carga el archivo
        ds = load_data(uploaded_file)

        # Confirmación carga archivo
        st.sidebar.write("**Se ha cargado un archivo.**")
        # st.sidebar.write(detect(uploaded_file))

        # Se extraen los nombres de las columnas del dataset cargado.
        columns_names_list = ds.columns.to_list()
        # st.sidebar.write(columns_names_list)
        with st.sidebar:
        # Widget para seleccionar las variables monitoreadas a analizar.
            st.write(
                """
            ### 2️⃣ Seleccione los nombres de las columnas que contienen características
            """)
            # Selección de características, por defecto todas menos la última (probable target)
            selected_features = st.sidebar.multiselect(
                " Seleccione las características",
                columns_names_list, columns_names_list[:-1],
                help = "Debe seleccionar las características correspondientes a información relevante para el entrenamiento de los modelos, pueden ser tanto categóricas como numéricas. Por defecto se seleccionan todas las columnas a excepción de la última (posible target) "
            )

            # Widget de consulta si el dataset contiene etiquetas.
            labeled = st.selectbox(
                "¿El dataset posee etiquetas?",
                ["Seleccione una opción ✔️", "Sí", "No"],
                help = "Esta pregunta se refiere si la base de datos cargada contiene una columna con la información si los datos han sido etiquetados previamente como datos normales y anómalos.",
            )

            if labeled == "Sí":
                target = st.selectbox(
                    "Ingrese el nombre de la columna que contiene las etiquetas.",
                    columns_names_list,
                    help="Esta columna debe ser de tipo binario. Donde 0 corresponde a un dato normal y 1 a una medición anómala.",
                    index=len(columns_names_list) - 1
                )

            elif labeled == "Seleccione una opción✅":
                st.write("Las preguntas anteriores son obligatorias.")

            #Etiqueta para comenzar
            ready = False
            if labeled != "Seleccione una opción ✔️":
                if labeled == 'Sí':
                    selected_df = ds[selected_features]
                    selected_df['Etiqueta'] = ds[target]
              
                ready = st.button("Comenzar!")
        st.write('## Análisis exploratorio estadístico y visual de los datos cargados: ')
        with st.expander("🕵️ Reporte exploratorio preliminar 📃", expanded=True):
                
            a1, a2 = st.columns([2,3])
            if labeled == 'Sí':

                a1.subheader('Set de datos original:')
                a1.dataframe(selected_df)  # visualización del dataset original (completo)
                
                describe=selected_df.describe().T.style.bar(subset=['mean'], color='#E68193')\
                        .background_gradient(subset=['std'], cmap='mako_r')\
                            .background_gradient(subset=['50%'], cmap='mako')
                a2.subheader("Descripción estadística de los datos cargados")
                a2.dataframe(describe) #descripción estadística columna 2

                #creación nuevo dataframe para el cambio de etiquetas (manual)
                df=pd.DataFrame()
                df['etiqueta conjunta'] = selected_df['Etiqueta'].replace([0,1],['normal','anomalía'])
                d = pd.DataFrame(df['etiqueta conjunta'].value_counts())
                #Gráfico torta composición etiquetas
                fig = px.pie(d,values='etiqueta conjunta',names=['normal','anomalía'],hole=0.4,opacity=0.6,
                            color_discrete_sequence=[colors_blue[3],colors_green[3]],
                            labels={'label':'etiqueta conjunta','etiqueta conjunta':'No. Of Samples'})

                fig.add_annotation(text='Los resultados sugieren un set de datos desbalanceados',
                                x=1.3,y=0.9,showarrow=False,font_size=18,opacity=0.7,font_family='monospace')
                fig.add_annotation(text='Etiquetado <br> Experto',
                                x=0.5,y=0.5,showarrow=False,font_size=14,opacity=0.7,font_family='monospace')

                fig.update_layout(
                    font_family='monospace',
                    title=dict(text='¿Cuántos datos corresponden a datos normales?',x=0.47,y=0.98,
                            font=dict(color=colors_dark[2],size=28)),
                    legend=dict(x=0.37,y=-0.05,orientation='h',traceorder='reversed'),
                    hoverlabel=dict(bgcolor='white'))

                fig.update_traces(textposition='outside', textinfo='percent+label')
                st.subheader('Composición de etiquetas:')
                st.plotly_chart(fig, use_container_width=True)
        
                st.subheader('Distribuciones de las características:')
                selected_df['Etiqueta'].replace([0,1],['normal','anomalía'],inplace=True)
                for label in selected_features:
                    f = generar_distrib(selected_df,label)

                    st.plotly_chart(f, use_container_width=True)
                
            pr_button=False
            if st.button("Generar un reporte exploratorio más detallado 🕵️"):

                pr = selected_df.profile_report()
                # profile = ProfileReport(pr, title="Reporte de exploración de datos")
                pr_button=True
           
            if pr_button==True:
                st_profile_report(pr)

        if ready:
                      
            # if st.button("Generar un reporte exploratorio inicial 🕵️"):

            # if st.button('Generar reporte'):
            #     
            #         time.sleep(3)
                    
            # %% 

            st.write('## Detección de anomalías')

            # if st.button('Entrenar modelos '):
            with st.spinner('Entrenando los modelos, esto puede tardar unos minutos...'):
                antes = time.time()

                best, grid1, grid2 = entrenar_modelos(selected_df, 'Etiqueta', 'F1')

                despues = time.time()
                delta_t = despues - antes
                str_t = 'El entrenamiento demoró: ' + str(delta_t) + ' segundos.'
                st.write(str_t)
                    # pycaret_s = setup(complete_df, target = 'target', session_id = 123, silent = True, use_gpu = True, profile = False)
                # model training and selection
                # best = compare_models(sort='F1')#,n_select=3)
                # score_grid = pull()
                st.write('### Grilla de búsqueda de modelos:')
                st.write(grid1.sort_values('F1',ascending=False).style.background_gradient(axis=0,cmap='mako'))
                # st.write(grid2)

                st.write('### Apilamiento de los mejors 5 modelos con mejor desempeño:')
                st.write('# Los mejores clasificador fueron:')
                st.write(grid1.sort_values('F1',ascending=False).head(5).style.background_gradient(axis=0,cmap='mako'))
                # st.write(supervised.pull())

                metrics = supervised.get_metrics()
                print(metrics)
                # st.write(type(metrics))

                
                # supervised.evaluate_model(best)
                
                supervised.plot_model(best, plot='confusion_matrix', display_format='streamlit',
                                    plot_kwargs={'percent': True})
                supervised.plot_model(best, plot='class_report', display_format='streamlit')

                pred_holdout = supervised.predict_model(best)
                st.write('La salida de los modelos fue:')
                st.dataframe(pred_holdout)
                supervised.plot_model(best, plot='error', display_format='streamlit')
                supervised.plot_model(best, plot='pr', display_format='streamlit')
                # supervised.plot_model(best, plot='boundary', display_format='streamlit')
                # supervised.plot_model(best,plot = 'feature',display_format='streamlit')
                # leaderboard = supervised.get_leaderboard()
                # st.experimental_show(leaderboard)
                
                supervised.interpret_model(best)
            
                # supervised.plot_model(best,plot = 'parameter',display_format='streamlit')
                # supervised.plot_model(best, plot='pipeline', display_format='streamlit')

                # supervised.plot_model(best, plot='calibration', display_format='streamlit')   
                # supervised.plot_model(best,plot = 'vc',display_format='streamlit')                             

except KeyError:
    st.error("Se ha ingresado un archivo sin la sintaxis pedida.", icon = "📄")

except ValueError:
    st.error("Oops, algo salió mal. Por favor comprueba que los valores ingresados seas consistentes. (ValueError).", icon = "🚨")

except TypeError:
    st.error("Oops, algo salió mal. Por favor comprueba que no las opciones ingresadas por incosistencias (TypeError).", icon = "⚠️")
   