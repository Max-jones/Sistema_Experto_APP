# %%%% 
from timeit import timeit
import pandas as pd
# raiz_ntbk = 'C:\\Users\\elmha\OneDrive - Universidad de Chile\GitHub\Sistema_Experto_APP\data\\artificiales\'
raiz = "C:\\Users\\Universidad de Chile\Documents\\GitHub\\Sistema_Experto_APP\\data\\artificiales\\"
# %%%% 
# %timeit
p1 = pd.read_csv(raiz+'Pozo_1_Labeled.csv',sep=';')

p2 = pd.read_csv(raiz+'Pozo_2_Labeled.csv',sep=';')

p3 = pd.read_csv(raiz+'Pozo_3_Labeled.csv',sep=';')


p = [p1,p2,p3]
i=1
for pozo in p:
    print('procesando pozo ',i)
    pozo['Date_Time'] = pozo['Date'] + ' ' + pozo['Time']
    pozo.drop(columns=['Date','Time'],inplace=True)# %%
    name = 'pozo_'+str(i)+'_procesado.csv'
    print(name)
    pozo["Date_Time"] = pd.to_datetime(pozo["Date_Time"])
    pozo.set_index("Date_Time", inplace=True)
    # print(pozo)
    # pozo.apply(lambda x: str(x.replace(',','.')))
    pozo.replace(',','.',regex=True,inplace=True)
    pozo.to_csv(name,sep=';')

    i+=1