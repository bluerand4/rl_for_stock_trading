#%%
from import_basics import *
# %%
path1='/Users/mac1/Documents/data/RL/'
data=os.listdir(path1)
data
# %%
i=32

df=read_excel(os.path.join(path1,data[i]))
df

pnl=df['pnl'].values.tolist()
import matplotlib.pyplot as plt 

plt.plot(pnl)
plt.ylabel('pnl')
plt.show()# %%
#%%
sum1=0
for i in range(len(data)):
    df=read_excel(os.path.join(path1,data[i]))
    df

    pnl=df['pnl'].values.tolist()

    sum1+=np.mean(pnl)
#%%
sum1
#%%
name1=''
# pnl=df[name1].values.tolist()
import matplotlib.pyplot as plt

# plt.plot(df['a0'].values.tolist())
# plt.plot(df['a1'].values.tolist())
plt.plot(df['a0'].values.tolist())
plt.ylabel('HOLD')
plt.show()# %%
#%%

#%%
df.iloc[0:2]
# %%
df
# %%
import matplotlib.pyplot as plt

plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()
plt.plot(hist.history['val_accuracy'],label='val_accuracy')
plt.plot(hist.history['accuracy'],label='accuracy')
plt.legend()
plt.show()