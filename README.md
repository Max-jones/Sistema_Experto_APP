# Sistema_Experto_APP 馃殌
Repositorio de aplicaci贸n web para el sistema experto

#Ejemplo
![image](https://user-images.githubusercontent.com/29563000/122610612-ee0fbc80-d04d-11eb-8d43-a8692e3455c4.png)

<!-- ## Requiere Python 3
y -->
## Dependencias:
```
streamlit==0.83
pandas
numpy
altair
pydeck
bokeh
plotly
```
## Instrucciones

### Instalaci贸n
Requiere Python 3 馃悕

Para utilizar con un virtualenv (recomendado)
https://whiteboxml.com/blog/the-definitive-guide-to-python-virtual-environments-with-conda
1. cd to the directory where requirements.txt is located
2. activate your virtualenv
3. run:

```console
pip install -r requirements.txt
```
 in your shell.
 
Y si no, en una terminal que tenga acceso a pip, ejecutar el comando 
```console
$ pip install -r requirements.txt'
```

### Correr la aplicaci贸n de forma local.

Abriendo un terminal con el virtualenv o si se realiz贸 la instalaci贸n de la segunda forma (debe estar instalada la librer铆a streamlit)

Para poder probar la app es necesario abrir un terminal en la carpeta del proyecto y correr

```console
streamlit run streamlit_app.py
```

### Correr la aplicaci贸n WEB.

1. Dirigirse a [Enlace Aplicaci贸n](https://share.streamlit.io/max-jones/sistema_experto_app/main/app.py)

2. Cargar un archivo .csv con la estructura de data del proyecto.

![DEMO](images/demo.gif)
