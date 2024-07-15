#%%

from constraint import Problem, AllDifferentConstraint

def stock_trading_csp(budget, prices, max_stocks):
    # Create a CSP problem
    problem = Problem()

    # Defining the stocks and their possible actions
    stocks = ['Stock1', 'Stock2', 'Stock3', 'Stock4']
    actions = ['Buy', 'Sell', 'Hold']  # Actions available for each stock
    
    # Add variables to the problem (each stock can have any of the specified actions)
    for stock in stocks:
        problem.addVariable(stock, actions)

    # Define a budget constraint function
    def budget_constraint(*actions):
        cost = sum(prices[stock] if action == 'Buy' else 0 for stock, action in zip(stocks, actions))
        return cost <= budget

    # Add budget constraint
    problem.addConstraint(budget_constraint, stocks)
    
    # Ensure only up to 'max_stocks' are bought
    def buy_limit_constraint(*actions):
        return actions.count('Buy') <= max_stocks
    
    # Add buying limit constraint
    problem.addConstraint(buy_limit_constraint, stocks)

    # Optional: All stocks must have different actions
    # problem.addConstraint(AllDifferentConstraint(), stocks)

    # Find and return all solutions
    return problem.getSolutions()

# Prices for each stock
stock_prices = {'Stock1': 50, 'Stock2': 60, 'Stock3': 55, 'Stock4': 45}

# Example usage of the CSP for stock trading
budget = 150  # Total budget to spend
max_to_buy = 2  # Maximum number of stocks to buy

solutions = stock_trading_csp(budget, stock_prices, max_to_buy)
for i, solution in enumerate(solutions):
    print(f"Solution {i + 1}: {solution}")

# %%
#%%
#%%
import pandas as pd
import numpy as np
from constraint import Problem

# Data Preparation
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

# CSP Setup
problem = Problem()

# Defining budget and stock constraints
budget_per_day = 300  # Total budget to spend per trading decision
max_stocks_per_trade = 3  # Maximum number of stocks that can be bought/sold

# Actions can be the number of stocks bought/sold (positive for buy, negative for sell, 0 for hold)
actions = range(-max_stocks_per_trade, max_stocks_per_trade + 1)

# Add variables (one per day) with their possible actions
for index, row in df.iterrows():
    problem.addVariable('Day_{}'.format(index), actions)

# Constraint: Ensure the total spending does not exceed the budget
def budget_constraint(*decision):
    total_cost = sum(decision[i] * df.loc[i, 'Close'] if decision[i] > 0 else 0 for i in range(len(decision)))
    return total_cost <= budget_per_day

problem.addConstraint(budget_constraint, ['Day_{}'.format(i) for i in range(len(df))])

# Solve the CSP
solutions = problem.getSolutions()

# Display a solution if available
if solutions:
    print("One possible set of trading decisions:")
    for key, value in solutions[0].items():
        print(f"{key}: {'Buy' if value > 0 else 'Sell' if value < 0 else 'Hold'} {abs(value)} stocks")
else:
    print("No solution found given the constraints.")

# %%
import pandas as pd
import numpy as np
from constraint import Problem

# Data Preparation
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

# CSP Setup
problem = Problem()

# Defining budget and stock constraints
budget_per_day = 300  # Total budget to spend per trading decision

# Actions and Variables Setup
for index, row in df.iterrows():
    max_stocks = budget_per_day // row['Close']  # Maximum stocks that can be bought given the budget
    actions = range(-max_stocks, max_stocks + 1)  # Possible actions from selling up to buying
    problem.addVariable('Day_{}'.format(index), actions)

# Constraint: Ensure the spending each day does not exceed the budget
def daily_budget_constraint(*decisions):
    return all((df.loc[i, 'Close'] * abs(decisions[i]) <= budget_per_day) for i in range(len(decisions)))

problem.addConstraint(daily_budget_constraint, ['Day_{}'.format(i) for i in range(len(df))])

# Solve the CSP
solutions = problem.getSolutions()

# Display a solution if available
if solutions:
    print("One possible set of trading decisions:")
    for key, value in solutions[0].items():
        action_type = 'Buy' if value > 0 else 'Sell' if value < 0 else 'Hold'
        print(f"{key}: {action_type} {abs(value)} stocks")
else:
    print("No solution found given the constraints.")

# %%
