#%%
from import_basics import *


# %%
ticker='AAPL'
fullname=generate_fullname_tradingview(ticker)
minute1='5'
df0=tradingview_simple(fullname,minute1)

# %%
df0=feature_engineering(df0)

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s):
        
        x1 = self.fc1(s)
        x1 = self.fc1_1(x1)
        x1= self.fc2(x1)
        x1=self.softmax(x1)
        
        return x1
    
# %%
df=copy.deepcopy(df0)
df['Date'] = pd.to_datetime(df['Date'])



# Extract day of week as categorical numbers (Monday=0, Sunday=6)

df['Day'] = df['Date'].dt.dayofweek



# Extract time

df['Time'] = df['Date'].dt.time


df
#%%

#%%
columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume','SMA50','SMA200','SMA1000','SMA2000']
for column in columns_to_normalize:
    df[column] = (df[column] - df[column].mean()) / df[column].std()
df

#%%
time_reference = pd.Timedelta(hours=4)  # Reference time of 4:00
df['MinutesSince4'] = (df['Date'].dt.time.apply(
    lambda x: pd.Timedelta(hours=x.hour, minutes=x.minute)) - time_reference).dt.total_seconds() / 60

# Categorize into 5-minute intervals (0, 1, 2, ...)
df['TimeCategory'] = (df['MinutesSince4'] // 5).astype(int)

df[['Date', 'Time', 'MinutesSince4', 'TimeCategory']]
#%%
df['Day']=df['Day']/df['Day'].max()
df['TimeCategory']=df['TimeCategory']/df['TimeCategory'].max()
#%%
df1=df[columns_to_normalize+[ 'Day', 'TimeCategory']]
df1
#%%
#%%
BACKS=5
device='cpu'
GAMMA=0.999
ALPHA=0.002

i=BACKS+2
position=0
pnl=0
stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
s=stock_data+[position]+[pnl]
s=torch.tensor(s).to(device)
s
#%%
print("\n>> s.shape= ", s.shape)
#%%
(2+len(columns_to_normalize))*BACKS+2
#%%
model = SimpleLinearModel((2+len(columns_to_normalize))*BACKS+2, 5).to(device)
dict1 = defaultdict(lambda: deque(maxlen=1000))
optimizer = torch.optim.Adam(model.parameters())


for _ in range(500):
    i=BACKS+2
    position=0
    pnl=0
    stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
    s=stock_data+[position]+[pnl]
    s=torch.tensor(s).to(device)
    s
    s.shape

    s.dtype

    done=False
    all_list=[]
    immediate_list=[]
    action_list=[]
    while i<len(df)-1:
        
        


        prev_pnl=copy.deepcopy(pnl)
        prev_position=copy.deepcopy(position)

        probs=model(s.to(device))
        probs

        probs

        a=torch.distributions.Categorical(probs).sample()
        a

        a=a.item()


        if a==2:
            position-=1
        elif a==1:
            position+=1

        price_now=df['Close'][i]
        price_now

        Gt=0

        i+=1

        stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
        price_next=df['Close'][i]
        price_next




        current_pnl=position*(price_next-price_now)
        current_pnl

        pnl=prev_pnl+current_pnl
        # print(i,"pnl: ",pnl)

        ns=stock_data+[position]+[pnl]
        ns=torch.tensor(ns,dtype=torch.float32)
        ns

        ns.dtype

        immediate=pnl
        immediate
        action_list.append(a)

        immediate_list.append(immediate)
        all_list.append([s,ns,a,immediate,done,Gt])


        s=copy.deepcopy(ns)
    

    max1=max(immediate_list)

    min1=min(immediate_list)
    a0=action_list.count(0)
    a1=action_list.count(1)
    a2=action_list.count(2)
    
    print(_,max1,min1,a0,a1,a2)



    Gt_list=[]
    for i ,(s,ns,a,immediate,done,_) in enumerate(all_list):
        Gt=0

        for ii,(s2,ns2,a2,im2,done2,_) in enumerate(all_list[i:]):
            im2=(im2-min1)/(max1-min1)
            Gt+=GAMMA**ii*im2
        Gt_list.append(Gt)
        all_list[i][-1]=Gt
    all_list


    gmax1=max(Gt_list)


    gmin1=min(Gt_list)


    for i ,(s,ns,a,immediate,done,Gt) in enumerate(all_list):
        optimizer.zero_grad()

        probs=model(s.to(device))
        Gt=(Gt-gmin1)/(gmax1-gmin1)
        loss=-torch.log(probs[a]) *Gt
        
        # Compute the gradients
        loss.backward()

        # Perform a single optimization step
        optimizer.step()

# %%
# %%
