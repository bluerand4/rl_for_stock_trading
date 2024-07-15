#%%

"""
1. MC method
2. it is great. i can achieve goal in 300 epochs.
3. this is PPO finally

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

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, s):
        x1=x(s)
        x1 = self.fc1(x1)
        x1 = F.relu(self.fc1_1(x1))
        x1= self.fc2(x1)
        # x1=self.softmax(x1)
        
        return x1
    
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s):
        x1=x(s)
        x1 = self.fc1(x1)
        x1 = F.relu(self.fc1_1(x1))
        x1= self.fc2(x1)
        x1=self.softmax(x1)
        
        return x1

class Actor_Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 4)
        self.fc3 = nn.Linear(output_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,s):
        x1=x(s)
        x1 = self.fc1(x1)
        x1 = F.relu(self.fc1_1(x1))
        x2= self.fc2(x1)
        probs=self.softmax(x2)
        value= self.fc3(x1)
        return probs,value
    

SIZE=3

GAMMA=0.7
ALPHA=0.02
EPSILON=0.2
# actor = Actor(36, 5)
# critic= Critic(36,5)
ac=Actor_Critic(36,5)
actor=Actor(36,5)
critic=Critic(36,5)
dict1 = defaultdict(lambda: deque(maxlen=1000))
#%%
for _ in range(300):
    s=starting_state()
    done=False
    all_list=[]

    while not done:
        # probs,v_hat=ac(s)
        probs=actor(s)
        v_hat=critic(s)
        a=torch.distributions.Categorical(probs).sample()
        ns=next_state(s,a)
        prob_old=probs[a].item()
        

        immediate,done=check_immediate_and_done(ns)
        
        
        # v=critic(s)
        if done:
            nv_hat=0
        else:
            # probs2,nv_hat=ac(ns)
            nv_hat=critic(ns)
        
        Gt=0
        A=0
        all_list.append([s,ns,a,immediate,done,v_hat,prob_old,Gt,A])

        s=copy.deepcopy(ns)
    all_list


    # p3 just update Gt
    for i ,(s,ns,a,immediate,done,v_hat,prob_old,Gt,_) in enumerate(all_list):
        Gt=0
        for ii,(s2,ns2,a2,im2,done2,v_hat2,prob_old2,Gt2,_) in enumerate(all_list[i:]):
            Gt+=GAMMA**ii*im2
            print("Gt: ",Gt)
        all_list[i][-2]=Gt

        # Q=immediate+GAMMA*nv_hat
        Q=Gt
        V=v_hat
        
        A=Q-V
        all_list[i][-1]=A
        
    all_list


    # p4 loss backward for actor, critic
    actor_loss=0
    critic_loss=0
    for i ,(s,ns,a,immediate,done,v_hat,prob_old,Gt,A) in enumerate(all_list):
        
        # p5 for actor
        # probs,v_hat=ac(s)
        probs=actor(s)
        v_hat=critic(s)

        s1= (probs[a]/prob_old)*A
        s2= torch.clamp((probs[a]/prob_old),1- EPSILON, 1+EPSILON) * A

        # actor_loss+=-torch.log(probs[a]) *Gt
        
        actor_loss-= torch.min(s1,s2)
        # actor_loss+=-torch.log(probs[a]) *A
        print("v_hat: ",v_hat)
        print("Gt: ",Gt)
        critic_loss+=(Gt-v_hat)**2
        # F.mse_loss(v_hat,torch.tensor([Gt],dtype=torch.float))
        # critic_loss+=F.mse_loss(v_hat,torch.tensor([Gt],dtype=torch.float))
        # critic_loss=ALPHA2*critic_loss
        
    actor_loss=actor_loss/len(all_list)
    critic_loss=critic_loss/len(all_list)
    # loss=actor_loss+critic_loss
    # loss.backward(retain_graph=True)
    actor.zero_grad()
    actor_loss.backward()

        # p7 update actor
    with torch.no_grad():
        for param in actor.parameters():
            param-=ALPHA*param.grad/len(all_list)
            # print("param.grad: ",param.grad)
    actor.zero_grad()

    critic.zero_grad()
    critic_loss.backward()
    with torch.no_grad():
        for param in critic.parameters():
            param-=ALPHA*param.grad/len(all_list)
            # print("param.grad: ",param.grad)
    critic.zero_grad()

    
    



#%%
s=starting_state()
done=False
all_list=[]

for _ in range(4):
    # probs,value=ac(s)
    probs=actor(s)
    value=critic(s)
    print("value: ",value)
    print("probs: ",probs)

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
    if ns==(2,2):
        print('win!')
        break
    if done:
        
        
            raise TypeError()
    all_list.append([s,ns,a,immediate,done,Gt])
    s=copy.deepcopy(ns)
# %%
