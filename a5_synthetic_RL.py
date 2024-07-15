#%%
import torch
import matplotlib.pyplot as plt

def generate_synthetic_stock_data(initial_price, mu, sigma, num_steps, time_step=1.0):
    """
    Generates synthetic stock price data using Geometric Brownian Motion.
    
    Args:
    initial_price (float): The initial stock price.
    mu (float): The drift (mean) of the stock price.
    sigma (float): The volatility (standard deviation) of the stock price.
    num_steps (int): The number of time steps to simulate.
    time_step (float): The time increment for each step, default is 1.0.
    
    

    Returns:
    torch.Tensor: A tensor containing the synthetic stock prices.
    """
    # Initialize the tensor for stock prices
    stock_prices = torch.zeros(num_steps)
    stock_prices[0] = initial_price
    
    # Generate random shocks
    random_shocks = torch.randn(num_steps - 1)
    
    for t in range(1, num_steps):
        dt = time_step
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * random_shocks[t - 1] * torch.sqrt(torch.tensor(dt))
        stock_prices[t] = stock_prices[t - 1] * torch.exp(drift + diffusion)
    
    return stock_prices

# Example usage
initial_price = 100.0  # Starting price of the stock
mu = 0.001  # Mean return
sigma = 0.03  # Volatility
num_steps = 10000  # Number of time steps to simulate

synthetic_data = generate_synthetic_stock_data(initial_price, mu, sigma, num_steps)

# Plot the synthetic stock price data
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data.numpy(), label='Synthetic Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.title('Synthetic Stock Price Data (Geometric Brownian Motion)')
plt.legend()
plt.grid(True)
plt.show()

# %%
