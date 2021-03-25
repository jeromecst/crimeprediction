import pandas as pd
pd.set_option('display.width', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-bright','classic'])

data = pd.read_csv('CrimesHalf.csv', low_memory=False)
print(data.head(100))
print(data.columns)
