import json
# import pandas as pd

path = "C:\\Users\\elmha\\OneDrive - Universidad de Chile\\GitHub\\Sistema_Experto_APP\\myfile.json"

_json = json.load(open(path))

df = pd.DataFrame(_json['series'][0]['values'],columns=_json['series'][0]['columns'])