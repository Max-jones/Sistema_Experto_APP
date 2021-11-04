import pandas as pd
import pytz

local_pathh = 'C:\\Users\\elmha\\OneDrive - Universidad de Chile\\GitHub\\Sistema_Experto_APP\\data\Horcon-etiquetado_con_1_etiqueta.csv'
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


