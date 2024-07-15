#%%
from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil,ast,subprocess,requests
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt


def reset_index(df):
    df=df.reset_index()    
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])
    return df

def read_excel(path):
    if path.endswith(".csv"):

        df=pd.read_csv(path,index_col=0)
    else:
        df=pd.read_excel(path,index_col=0)
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])    
    return df


def dump1(*varnames):
    python_script=os.path.basename(__file__)
    if not os.path.exists(f't7_print/{python_script}'):
        os.makedirs(f't7_print/{python_script}')
    global_vars = globals()
    string1=''
    for varname in varnames:
        if varname in global_vars:
            dict1={varname: global_vars[varname]}
            string1=string1+str(dict1)+', '
    print(string1)
    filename=string1
    with open(f't7_print/{python_script}/{filename}','w') as f:
        f.write(string1)

def print1(content_dict,filename):
    if '.txt' not in filename:
        filename=filename+'.txt'
    path='data/log/'
    if not os.path.exists(path):
        os.makedirs(path)
    fullpath=path+filename
    timedelta1=0
    timedelta2=100
    script_name = os.path.basename(__file__)
    today1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1)).strftime("%Y-%m-%d %H:%M:%S"))
    today2=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1))).timestamp()
    past1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta2)).strftime("%Y-%m-%d"))
    past2=datetime.strptime('20210218', "%Y%m%d").timestamp()

    with open(fullpath,'a+') as f:
        f.write(str(content_dict))
        f.write(f' <-- {today1} {script_name} -->')   
        f.write('\n') 



def list_minus(list1, list2):
    # Convert lists to sets and find the difference
    set_difference = set(list1) - set(list2)
    # Convert the set back to a list
    return list(set_difference)



def base_list_save(filename,list1):
    
    content=str(list1)

    if filename.endswith('txt'):
        filename=filename.replace('.txt','')
    path='log/lists'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'log/lists/{filename}.txt','w') as f:
        f.write(content)

def base_list_collections():
    return os.listdir('log/lists')

def base_list_read(filename):
    with open(f'log/lists/{filename}.txt','r') as f:
        data=f.readline()
    data=ast.literal_eval(data)
    return data



def csv_update_insert_one(database_name,collection_name,dn,no_duplicate_column):
    

    # csv_update_insert_one
    if 'cjs' in getpass.getuser():
        path3='C:/Users/cjsdl/Documents/data/'
    elif 'linux' in getpass.getuser():
        path3=f'/home/{getpass.getuser()}/Documents/data/'
    else:
        path3=f'/Users/{getpass.getuser()}/Documents/data/'
    
    path=path3+database_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)



    fullpath=path+collection_name+'.csv'
    try:
        do=read_excel(fullpath)
        
        do.set_index(no_duplicate_column, inplace=True)
        dn.set_index(no_duplicate_column, inplace=True)
        # print("dn: ",dn)
        # Update 'do' with the values from 'dn' where the indices match, then concatenate the rest of 'dn'
        do.update(dn)
        do = pd.concat([do, dn[~dn.index.isin(do.index)]])

        # Reset the index if needed
        do.reset_index(inplace=True)
        # print("do: ",do)        
        do

    except:
        # do=pd.DataFrame()
        do=dn

    do.to_csv(fullpath)
    # print('success csv')
    return do


def open_excel(df):
    random1=random.randint(1,10)

    temp_file=f'temp_sample_{random1}.xlsx'
    if 'cjsdl' in getpass.getuser():
        export_path='G:/My Drive/ibkr/1_total_data/8 temp/'
        print('start - to_excel a large file maybe ...')
        df.to_excel(export_path+temp_file)
        print('success - to_excel a large file maybe ...')
        excel1=fr"G:\My Drive\ibkr\1_total_data\8 temp\temp_sample_{random1}.xlsx"
        # r"C:\Users\cjsdl\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\Anaconda Prompt (miniconda3).lnk"
        os.startfile(excel1)
        
    else:
        export_path=f'/Users/{getpass.getuser()}/Library/CloudStorage/GoogleDrive-ryan.ichun9@gmail.com/My Drive/ibkr/1_total_data/8 temp/'
        print('start - to_excel a large file maybe ...')

        df.to_excel(export_path+temp_file)
        os.system("open -a 'Microsoft Excel.app' '{}{}'".format(export_path,temp_file))
# %%

from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict, deque



def find_exchange_v2(ticker):


    url = f"https://finance.yahoo.com/quote/{ticker}"

    querystring = {"p":ticker}

    payload = ""
    headers = {"User-Agent": "insomnia/8.5.0"}
    
    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)

    # print(response.text)
    # //div[@class="C($tertiaryColor) Fz(12px)"]
    TAG='div'
    class_name = "C($tertiaryColor) Fz(12px)"

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.text

    elements = soup.find_all(TAG, class_=class_name)
    exchange_info=elements[0].find('span').text
    exchange_info=exchange_info.lower()

    exchange='NYSE'
    if exchange_info.startswith('nasdaq'):
        exchange='NASDAQ'
    elif exchange_info.startswith('nyse'):
        exchange='NYSE'
    return exchange



def find_exchange_v3(ticker):
    print("ticker: ",ticker)
    #https://polygon.io/docs/stocks/get_v3_reference_tickers__ticker
    import requests,json
    url=f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={polygonkey}'
    response=requests.get(url)

    json1=response.json()
    stock_name=json1['results']

    exchange1=stock_name['primary_exchange']
    exchange1

    if exchange1=='XNAS':
        exchange='NASDAQ'
    elif exchange1=='XNYS':
        exchange='NYSE'
    else:
        exchange='AMEX'
    return exchange
#%%
def generate_fullname_tradingview(ticker):
    try:
        exchange=find_exchange_v3(ticker)
    except:
        exchange=find_exchange_v2(ticker)
    return f'{exchange}:{ticker}'
from websocket import create_connection


def datainput(func,paramList):
    #func="set_auth_token"
    #paramList=["unauthorized_user_token"]
    data1=json.dumps({"m":func,"p":paramList}, separators=(',', ':'))
    data1
    data2="~m~" + str(len(data1)) + "~m~" + data1
    return data2

import json
import json,string,re

def tradingview_simple(fullname,minute1):

    if "CME" in fullname or "CBOT" in fullname or "COMEX" in fullname or "NYMEX" in fullname:
        extended_or_regular="regular"
    else:
        extended_or_regular="extended"


    headers = json.dumps({'Origin': 'https://data.tradingview.com'})
    connector=create_connection('wss://data.tradingview.com/socket.io/websocket',headers=headers)
    stringLength=12
    letters = string.ascii_lowercase
    random_string= ''.join(random.choice(letters) for i in range(stringLength))
    session= "qs_" +random_string

    stringLength=12
    letters = string.ascii_lowercase
    random_string= ''.join(random.choice(letters) for i in range(stringLength))
    chart_session= "cs_" +random_string
    #chart_session= "cs_" +"ZHlTeGX28izS"
    #ZHlTeGX28izS

    #connector.send(datainput('set_auth_token',["eyJhbGciOiJSUzUxMiIsImtpZCI6IkdaeFUiLCJ0eXAiOiJKV1QifQ.eyJ1c2VyX2lkIjoxMjgyNzE5MywiZXhwIjoxNjQ3OTM2MDQ2LCJpYXQiOjE2NDc5MjE2NDYsInBsYW4iOiJwcm9fcHJlbWl1bSIsImV4dF9ob3VycyI6MSwicGVybSI6ImNib3RfbWluaSxjbWUsY21lLWZ1bGwsY29tZXgsbnltZXgsbnlzZSxueW1leF9taW5pLGNtZV9taW5pLGNib3QsbmFzZGFxLGNvbWV4X21pbmksa3J4X3N0b2NrcyIsInN0dWR5X3Blcm0iOiJ0di12b2x1bWVieXByaWNlLHR2LXByb3N0dWRpZXMiLCJtYXhfc3R1ZGllcyI6MjUsIm1heF9mdW5kYW1lbnRhbHMiOjAsIm1heF9jaGFydHMiOjgsIm1heF9hY3RpdmVfYWxlcnRzIjo0MDAsIm1heF9zdHVkeV9vbl9zdHVkeSI6MjR9.ps5_xiYSdN_1FQnC1C98KwtTE3vHMRA0hDyHnCXbg3cyP0RHT6P4O4EBIGCeW-ZsrWkLrCdVN5ZJTgH9jfG9IaSviiDhkhS8rh6iFvqPS-NZ5rdcJ8--tbSLMcLcSHxfonrMO6BXB_yAh3fDePaWFFmJoaDh7D9_-iLgDC_jtoI"]))
    connector.send(datainput('set_auth_token',['unauthorized_user_token']))
    connector.send(datainput("chart_create_session",[chart_session, ""]))
    connector.send(datainput('quote_create_session',[session]))
    #connector.send(datainput('quote_set_fields',[session,"ch","chp","current_session","description","local_description","language","exchange","fractional","is_tradable","lp","lp_time","minmov","minmove2","original_name","pricescale","pro_name","short_name","type","update_mode","volume","currency_code","rchp","rtc"]))
    connector.send(datainput('quote_set_fields',[session,"short_name","pro_name","logoid","currency-logoid","base-currency-logoid","symbol-primaryname","type"]))
    "qs_pzepkyh89H2l"
    connector.send(datainput('quote_add_symbols',[session, fullname, "={\"symbol\":"+"\"{}\"".format(fullname) + ",\"currency-id\":\"USD\",\"adjustment\":\"splits\"}"]))

    connector.send(datainput('quote_fast_symbols',[session,fullname]))
    #connector.send(datainput('resolve_symbol',[chart_session,"sds_sym_1","={\"symbol\":"+"\"{}\"".format(fullname) +",\"adjustment\":\"splits\",\"session\":\"extended\"}"]))
    #connector.send(datainput('resolve_symbol',[chart_session,"sds_sym_1","={\"symbol\":"+"\"{}\"".format(fullname) +",\"adjustment\":\"splits\",\"session\":\"regular\"}"]))
    connector.send(datainput('resolve_symbol',[chart_session,"symbol_1","={\"symbol\":"+"\"{}\"".format(fullname) +",\"adjustment\":\"splits\",\"session\":" +"\"{}\"".format(extended_or_regular)+"}"]))
    connector.send(datainput('create_series',[chart_session, "s1", "s1", "symbol_1", minute1, 5000]))
    time1=[time.monotonic()]
    while True:
        #time.sleep(1)
        list1=[]
        result = connector.recv()
        list2=""+result+'\n'
        
        list1.append(result)
        #print(result)
        #print("helo##############################")

        if len(result)>10000:
            break
        elif len(result)==0:
            break
        else:
            
            if time.monotonic()-time1[0]>(60):
                print(fullname, "error here5")
                break

            else:
                continue
    result1=list1[0]
    a1=""+result1+'\n'
    out= re.search('"s":\[(.+?)\}\]', a1).group(1)
    x=out.split(',{\"')

    list4=[]

    for i in range(len(x)):


        xi=x[i]
        xi

        xi2= re.split('\[|:|,|\]', xi)
        

        index1= int(xi2[1])
        index1

        date2=xi2[4]
        date2

        date1=datetime.fromtimestamp(float(date2)).strftime("%Y/%m/%d, %H:%M:%S")
                
        date1=datetime.strptime(date1, "%Y/%m/%d, %H:%M:%S")
        date1

        list3=[index1,date1,float(xi2[5]), float(xi2[6]), float(xi2[7]), float(xi2[8]), float(xi2[9]),float(date2)]
        list3

        list4.append(list3)
        list4

    df=pd.DataFrame(list4)
    df.rename(columns={0:"Index",1:'Date',2:"Open",3:"High",4:"Low",5:"Close",6:"Volume",7:"Timestamp"}, inplace=True)


    df = df.drop('Index', axis=1)
    df

    while True:
        tot=len(df)-1
        date2=df['Date'][tot]
        date2
        if 'nan' in str(date2).lower() or 'nat' in str(date2).lower():
            print('drop last row')
            df = df.iloc[:-1 , :]
        else:
            break

    return df

import torch,os,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
import pandas as pd
if torch.backends.mps.is_available():
    device='mps'
elif torch.cuda.is_available():
    device='cuda'
else:
    device="cpu"

DEVICE=device

def random_seed(m):
    random.seed(m)
    torch.manual_seed(m)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(m)
        torch.cuda.manual_seed_all(m)
    np.random.seed(m)
# random_seed(1)




def feature_engineering(df):
    df['Close1']=df["Close"].shift(periods=1)

    df['Close5'] = df["Close"].rolling(min_periods=1, window=5).mean()
    df['Close10'] = df["Close"].rolling(min_periods=1, window=10).mean()
    df['SMA50'] = df["Close"].rolling(min_periods=1, window=50).mean()
    df['SMA200'] = df["Close"].rolling(min_periods=1, window=200).mean()
    df['SMA300'] = df["Close"].rolling(min_periods=1, window=300).mean()
    df['SMA500'] = df["Close"].rolling(min_periods=1, window=500).mean()
    df['SMA100'] = df["Close"].rolling(min_periods=1, window=100).mean()
    df['SMA1000'] = df["Close"].rolling(min_periods=1, window=1000).mean()
    df['SMA2000'] = df["Close"].rolling(min_periods=1, window=2000).mean()

    df["High5"] = df["High"].rolling(min_periods=1, window=5).max()
    df["High10"] = df["High"].rolling(min_periods=1, window=10).max()
    df["MAX200"] = df["High"].rolling(min_periods=1, window=200).max()

    df["Low5"] = df["Low"].rolling(min_periods=1, window=5).min()
    df["Low10"] = df["Low"].rolling(min_periods=1, window=10).min()
    df["MIN200"] = df["Low"].rolling(min_periods=1, window=200).min()
    

    df['Close2'] = df['Close'].pct_change()
    df['High2'] = df['High'].pct_change()
    df['Low2'] = df['Low'].pct_change()
    df['Open2'] = df['Open'].pct_change()
    df['Volume2'] = df['Volume'].pct_change()

    # p2 getting 'down2' value from 'low' and 2sd away from 'low'
    ROLLING_N=10
    df['Low5_mean'] = df['Low'].rolling(window=ROLLING_N).mean()
    df['Low5_sd'] = df['Low'].rolling(window=ROLLING_N).std()
    z_score=2
    df['up1'] = df['High'] + z_score * df['Low5_sd']
    df['down1'] = df['Low'] - z_score * df['Low5_sd']
    df
    df['down1']=df['down1'].shift(1)
    df['up1']=df['up1'].shift(1)
    
    return df