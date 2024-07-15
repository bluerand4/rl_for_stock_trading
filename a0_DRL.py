#%%

"""
nn
discrete space
using actor only. 
probability outputing 4x1


"""
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

PRINTABLE=False
def print5(*content):
    if PRINTABLE:
        print(*content)



SIZE=3

GAMMA=0.7
ALPHA=0.02

def next_state(s,a):
    row=s[0]
    col=s[1]
    if a == 0:  # up
        row, col = row - 1, col
        print5('up 0')
    elif a == 1:  # down
        row, col = row + 1, col
        print5('down 1')
    elif a == 2:  # left
        row, col = row, col - 1
        print5('left 2')
    elif a == 3:  # right
        row, col = row, col + 1
        print5('right 3')
    return (row,col)



def starting_state():
    return (0,0)
def check_immediate_and_done(s):
    row=s[0]
    col=s[1]
    done=False
    if row < 0 or col < 0 or row > 2 or col > 2:
        immediate = -10.0
        done = True
    elif row == 1 and col == 1:
        immediate = -.50
    elif row == 2 and col == 2:
        immediate = 10.0
        done = True
    elif row==2 and col==0:
        immediate=.01
    else:
        immediate = -.1
    print5(immediate,done)
    return immediate,done

def x(s):
    row=s[0]
    col=s[1]
    template=torch.zeros(3,3,4)
    template[row][col]=1
    template=template.flatten()
    # template=torch.tensor([row,col],dtype=torch.float)
    return template

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s):
        x1=x(s)
        x1 = self.fc1(x1)
        x1 = self.fc1_1(x1)
        x1= self.fc2(x1)
        x1=self.softmax(x1)
        
        return x1
    

model = SimpleLinearModel(36, 5)

dict1 = defaultdict(lambda: deque(maxlen=1000))

for _ in range(500):
    s=starting_state()
    done=False
    all_list=[]


    while not done:
        probs=model(s)
        a=torch.distributions.Categorical(probs).sample()
        ns=next_state(s,a)

        immediate,done=check_immediate_and_done(ns)

        Gt=0
        all_list.append([s,ns,a,immediate,done,Gt])

        s=copy.deepcopy(ns)
    all_list

    for i ,(s,ns,a,immediate,done,_) in enumerate(all_list):
        Gt=0
        for ii,(s2,ns2,a2,im2,done2,_) in enumerate(all_list[i:]):
            Gt+=GAMMA**ii*im2
        all_list[i][-1]=Gt
    all_list

    for i ,(s,ns,a,immediate,done,Gt) in enumerate(all_list):
        probs=model(s)
        loss=-torch.log(probs[a]) *Gt
        # loss.backward(retain_graph=True)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param-=ALPHA*param.grad
                print("param.grad: ",param.grad)
        model.zero_grad()
# %%

s=starting_state()
done=False
all_list=[]

for _ in range(4):
    probs=model(s)

    # print("probs: ",probs)

    # a=random.choices(range(4), weights=probs)[0]
    a=torch.argmax(probs).item()
    a

    ns=next_state(s,a)
    # print5("pi(s,a,h): ",pi(s,a,h))
    # 
    print("ns: ",ns)
    print("a: ",a)
    immediate,done=check_immediate_and_done(ns)
    print("immediate,done: ",immediate,done)
    Gt=0
    if done:
        raise TypeError()
    all_list.append([s,ns,a,immediate,done,Gt])
    s=copy.deepcopy(ns)
# %%
