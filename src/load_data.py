# load_data.py
import os
import pandas as pd
#  
import config

from load_local_data import load_data

# DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))[:-9] #without notebooks
# my_file = os.path.join(DATA_FOLDER, 'data/Horcon-etiquetado_con_1_etiqueta.csv')
# df = load_data(my_file)


otro=load_data(config.TRAINING_FILE)


# df.head()