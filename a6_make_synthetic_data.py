#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Parameters
mu = 0.001  # Expected return
base_sigma = 0.0  # Base volatility
MAG=0.09
oscillation_amplitude = MAG  # Amplitude of the oscillation
oscillation_frequency = MAG  # Frequency of the oscillation

def SDE_system(x, t):
    drift = mu  # Drift component models the expected return
    # Oscillating sigma
    sigma = base_sigma + oscillation_amplitude * np.sin(t)
    shock = sigma * np.random.normal()  # Shock component models volatility
    return drift + shock,sigma

T = 10000
dt = 0.015  # Time step size
x = torch.tensor([1.0])  # Initial stock price
stock_prices = []
times = np.linspace(0, T*dt, T)  # Generate time values
sigma_=[]
for i, t in enumerate(times):
    dx,sigma = SDE_system(x, t)
    x = x + dt * dx  # Euler's method for SDE
    stock_prices.append(x.item())
    sigma_.append(sigma)

# Plot the simulated stock prices
plt.figure(figsize=(10, 5))
plt.plot(times, stock_prices)
plt.title('Simulated Stock Price Movement with Oscillating Volatility')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# %%
plt.plot(sigma_)
plt.ylim(-1, 1)  # Set the y-axis limits from -10 to 10
plt.grid(True) 
plt.show()
# %%
np.sin(6)
# %%
plt.plot(stock_prices)

plt.show()
# %%
stock_prices
# %%
dn=pd.DataFrame(stock_prices)
dn.columns=['ticker']
dn
#%%
