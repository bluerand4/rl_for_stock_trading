#%%
import pandas as pd
import numpy as np

# Initialize DataFrame
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

# Determine actions based on MDP (simplified strategy)
def mdp_trading_strategy(df):
    inventory = 0
    capital = 1000  # Starting capital
    action_history = []
    profit_history = []

    for i in range(1, len(df)):
        current_price = df.loc[i, 'Close']
        previous_price = df.loc[i - 1, 'Close']
        action = 'Hold'

        if current_price < previous_price:  # Predict price will go up
            if capital >= current_price:
                action = 'Buy'
                inventory += 1
                capital -= current_price
        elif current_price > previous_price and inventory > 0:  # Predict price will go down
            action = 'Sell'
            inventory -= 1
            capital += current_price

        action_history.append(action)
        profit_history.append(capital + inventory * current_price - 1000)  # unrealized profit

    return action_history, profit_history

# Apply the strategy
actions, profits = mdp_trading_strategy(df)

# Display actions and profits
print("Actions taken:", actions)
print("Profit over time:", profits)

# %%
