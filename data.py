import pandas as pd
pd.set_option('display.width', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-bright','classic'])

data = pd.read_csv('Crimes1M.csv', low_memory=False)
del data['ID']
del data['Case Number']
del data['Updated On']
del data['Longitude']
del data['Latitude']
del data['Location']
del data['X Coordinate']
del data['Y Coordinate']
data = data.to_numpy() 
print("shape :",data.shape)
print("dates :", data[10:30,0])

def split_date(X):
    date = data[:,0]
    return 

split_date(data)
