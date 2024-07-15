#%%
import pandas as pd
import numpy as np
import random
# Create a simplified DataFrame (For example purposes)
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

def simulate_market_movement(price, action1, action2):
    """ Modify this function to use historical volatility instead of fixed percentages """
    if action1 == "Buy" and action2 == "Buy":
        return price * 1.02
    elif action1 == "Sell" and action2 == "Sell":
        return price * 0.98
    else:
        return price * np.random.choice([0.99, 1.01])

def evaluate_position(price, action, original_price):
    if action == "Buy":
        return original_price - price
    elif action == "Sell":
        return price - original_price
    return 0  # Hold

def minimax(price, depth, is_maximizing_player, actions, original_price, df_index):
    if depth == 0 or df_index >= len(df) - 1:
        return evaluate_position(price, 'Hold', original_price), 'Hold'

    if is_maximizing_player:
        best_value = -float('inf')
        best_action = None
        for action in actions:
            new_price = simulate_market_movement(price, action, random.choice(actions))
            eval, _ = minimax(new_price, depth - 1, False, actions, original_price, df_index + 1)
            if eval > best_value:
                best_value = eval
                best_action = action
        return best_value, best_action
    else:
        worst_value = float('inf')
        worst_action = None
        for action in actions:
            new_price = simulate_market_movement(price, random.choice(actions), action)
            eval, _ = minimax(new_price, depth - 1, True, actions, original_price, df_index + 1)
            if eval < worst_value:
                worst_value = eval
                worst_action = action
        return worst_value, worst_action

# Start adversarial search from the first index in the DataFrame
starting_price = df.loc[0, 'Close']
original_price = df.loc[0, 'Close']
actions = ['Buy', 'Sell', 'Hold']
best_value, best_action = minimax(starting_price, 3, True, actions, original_price, 0)
print(f"Best action: {best_action} with expected outcome: {best_value}")

# %%
df
# %%
