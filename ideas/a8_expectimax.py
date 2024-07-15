#%%
import pandas as pd
import numpy as np

# Generate DataFrame
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

def expectimax(df, index, holding, capital):
    if index >= len(df) - 1:
        return capital + holding * df.iloc[index]['Close'], []

    # Current close price
    current_price = df.iloc[index]['Close']
    
    # Actions: Buy, Sell, Hold
    # Buy
    if capital >= current_price:
        future_value_buy, path_buy = expectimax(df, index + 1, holding + 1, capital - current_price)
    else:
        future_value_buy, path_buy = float('-inf'), []
    
    # Sell
    if holding > 0:
        future_value_sell, path_sell = expectimax(df, index + 1, holding - 1, capital + current_price)
    else:
        future_value_sell, path_sell = float('-inf'), []

    # Hold
    future_value_hold, path_hold = expectimax(df, index + 1, holding, capital)

    # Choose the best option
    best_value = max(future_value_buy, future_value_sell, future_value_hold)
    if best_value == future_value_buy:
        return future_value_buy, ['Buy'] + path_buy
    elif best_value == future_value_sell:
        return future_value_sell, ['Sell'] + path_sell
    else:
        return future_value_hold, ['Hold'] + path_hold

# Starting parameters
initial_capital = 1000
initial_holding = 0

# Run expectimax
final_value, actions = expectimax(df, 0, initial_holding, initial_capital)
print("Final Portfolio Value:", final_value)
print("Actions Taken:", actions)

# %%
