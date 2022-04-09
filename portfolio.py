import numpy as np
import pandas as pd

from mean_variance_optimization import *

seed_everything()

NUM_PORTFOLIOS = 50000
stock_data = pd.read_csv("data/stock_prices.csv", parse_dates=True, index_col="Date")

def main():
    # Expected returns of the assets
    company_returns, covariance_matrix = stock_returns(stock_data)

    minimum_volatility = min_volatility(company_returns, covariance_matrix)

    maximum_sharpe_ratio = max_sharpe_ratio(company_returns, covariance_matrix)

    efficient_portfolio_return = efficient_return(company_returns, covariance_matrix)

    maximum_diversification = max_diversification(company_returns, covariance_matrix, long_only=True)

    monte_carlo_min_volatility, monte_carlo_sharpe_ratio = monte_carlo_optimisation(company_returns, covariance_matrix, NUM_PORTFOLIOS)

    columns = ['Asset allocation']+list(stock_data.columns)

    result = pd.DataFrame([['Minimum Volatility'] + minimum_volatility,
                          ['Maximum Sharpe Ratio'] + maximum_sharpe_ratio,
                          ['Efficien Portfolio Return'] + efficient_portfolio_return,
                          ['Maximum Diversification'] + maximum_diversification,
                          ['Monte Carlo Minimum Volatility'] + [round(x, 3) for x in monte_carlo_min_volatility][2:],
                          ['Monte Carlo Maximum Sharpe Ratio'] + [round(x, 3) for x in monte_carlo_sharpe_ratio][2:]],
                          columns=columns)
    print('Saving the result to ./result.csv')
    result.to_csv('./result.csv')
    # print(result)

if __name__ == "__main__":
    main()