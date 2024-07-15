#%%
import numpy as np
import pandas as pd
import random

# DataFrame setup
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

# State representation: Simple, using only 'Close' prices
states = df['Close'].values

# Q-table, rows are states (prices here), columns are actions (0=Buy, 1=Sell, 2=Hold)
Q = np.zeros((len(states), 3))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate

# For simplicity, simulate trading over the same data multiple times (epochs)
epochs = 10
stock_owned = 0
capital = 1000  # Initial capital, can buy at most one stock
profit = 0

for epoch in range(epochs):
    for i in range(len(states) - 1):
        current_state = i
        next_state = i + 1

        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 2)  # Explore action space
        else:
            action = np.argmax(Q[current_state])  # Exploit learned values

        # Simulate action and observe reward
        reward = 0
        if action == 0 and capital >= states[current_state]:  # Buy
            stock_owned += 1
            capital -= states[current_state]
        elif action == 1 and stock_owned > 0:  # Sell
            stock_owned -= 1
            capital += states[current_state]
            reward = states[current_state] - states[current_state - 1]  # Profit from last buy
        elif action == 2:  # Hold
            reward = 0  # No action taken

        # Q-learning update
        Q[current_state, action] = Q[current_state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[current_state, action])

# Results
print("Final Q-table:")
print(Q)

# Print out the final capital after trading
print("Final capital:", capital + stock_owned * states[-1])

# %%
