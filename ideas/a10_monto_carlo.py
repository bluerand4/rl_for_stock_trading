#%%
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame and contains the stock data
# Example DataFrame structure: df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
size=5000
data = {
    'Date': pd.date_range(start='1/1/2020', periods=size, freq='D'),
    'Open': np.random.randint(95, 105, size=size),
    'High': np.random.randint(105, 110, size=size),
    'Low': np.random.randint(90, 95, size=size),
    'Close': np.random.randint(95, 105, size=size),
    'Volume': np.random.randint(1000, 5000, size=size)
}
df = pd.DataFrame(data)
#%%
# df=pd.read_csv('data')
# df = pd.read_csv('data/AAPL_5min.txt', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], header=None)
df = pd.read_csv('data/BTC_5min.txt', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], header=None)
df


#%%

# Step 1: Calculate daily returns
df['Daily Return'] = df['Close'].pct_change()

# Remove any NaNs that might have occurred during the calculation
df = df.dropna()
df=df.reset_index()
#%%
df
#%%
# Step 2: Set up the Monte Carlo simulation
num_simulations = 1000  # Number of simulated paths
num_days = 252  # Number of trading days to simulate (typically 252 for a trading year)

last_price = df['Close'].iloc[-1]  # Starting price (most recent closing price)

# Resulting simulation array
simulation_df = pd.DataFrame()

# Loop through each simulation
for x in range(num_simulations):
    count = 0
    daily_volatility = df['Daily Return'].std()  # Standard deviation of returns
    
    price_series = []
    price = last_price * (1 + np.random.normal(0, daily_volatility))
    price_series.append(price)
    
    for y in range(num_days):
        if count == 251:
            break
        price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)
        count += 1
    
    simulation_df[x] = price_series

# Step 3: Review the results
# We can plot the simulations and calculate further statistics if needed
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(simulation_df)
plt.title('Monte Carlo Simulation of Stock Prices')
plt.ylabel('Price')
plt.xlabel('Days')
plt.show()

# %%
# Assuming 'simulation_df' is the DataFrame with simulation results as in the previous Python example
upper_bound_list = simulation_df.apply(lambda x: np.percentile(x, 95))
lower_bound_list = simulation_df.apply(lambda x: np.percentile(x, 5))
upper_bound=upper_bound_list.iloc[-1]
lower_bound=lower_bound_list.iloc[-1]
print("Upper Price Target for Risk Taking OR it means sell when it is above:", upper_bound)
print("Lower Price Target for Risk Management OR it means but when it is below:", lower_bound)



# %%

#%%
# Initialize variables
position = 0
pnl = 0
buy_price = 0
sell_price = 0

# Iterating through each day in the dataframe
for i in range(len(df)):
    Close = df['Close'][i]
    Open = df['Open'][i]
    
    # Check if there's no current position
    if position == 0:
        if Close < lower_bound:
            position += 1  # Take a long position
            buy_price = Close
            print(f"Bought at {buy_price} on day {i}")
    
    # Check if there is a current position
    elif position == 1:
        if Close > upper_bound:
            position -= 1  # Close the long position
            sell_price = Close
            pnl += sell_price - buy_price
            print(f"Sold at {sell_price} on day {i}, PnL: {sell_price - buy_price}")

if position==1:
    position -= 1  # Close the long position
    sell_price = Close
    pnl += sell_price - buy_price
    print(f"Sold at {sell_price} on day {i}, PnL: {sell_price - buy_price}")

# Output the total profit and loss
print(f"Total PnL: {pnl}")

# %%
# %%
