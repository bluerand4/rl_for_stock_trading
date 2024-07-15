#%%
from import_basics import *
# %%
df=pd.read_csv('data/AAPL_5min.txt',header=['d','o','h','l','c'])
df
# %%
import pandas as pd

# Corrected code to load the CSV file
df = pd.read_csv('data/AAPL_5min.txt', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], header=None)

# Display the DataFrame
print(df)

# %%
df
# %%
