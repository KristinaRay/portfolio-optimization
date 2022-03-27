import numpy as np
import pandas as pd

from mean_variance_optimization import *

seed_everything()

NUM_PORTFOLIOS = 50000
stock_data = pd.read_csv("data/stock_prices.csv", parse_dates=True, index_col="Date")


# Expected returns of the assets
company_returns, covariance_matrix = stock_returns(stock_data)

minimum_volatility = min_volatility(company_returns, covariance_matrix)
print(f"Shares in the minimum volatility portfolio \n{minimum_volatility}")

maximum_sharpe_ratio = max_sharpe_ratio(company_returns, covariance_matrix)
print(f"Shares in the maximum sharpe ratio portfolio \n{maximum_sharpe_ratio}")

efficient_portfolio_return = efficient_return(company_returns, covariance_matrix)
print(f"Shares in efficient return portfolio \n{efficient_portfolio_return}")

maximum_diversification = max_diversification(company_returns, covariance_matrix, long_only=True)
print(f"Shares in diversified portfolio \n{maximum_diversification}")

monte_carlo_min_volatility, monte_carlo_sharpe_ratio = monte_carlo_optimisation(company_returns, covariance_matrix, NUM_PORTFOLIOS)
print(f"Portfolio return with minimal volatility after Monte Carlo Optimization {monte_carlo_min_volatility['Returns']:.3f}")
print('Shares ', [round(x, 3) for x in monte_carlo_min_volatility])
maximum_sharpe_ratio = max_sharpe_ratio(company_returns, covariance_matrix)
print(f"Portfolio return with Maximum Sharpe ratio after Monte Carlo Optimization {monte_carlo_sharpe_ratio['Returns']:.3f}")
print('Shares ', [round(x, 3) for x in monte_carlo_sharpe_ratio])