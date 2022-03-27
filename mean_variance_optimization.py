
import random
import numpy as np
import pandas as pd
from scipy import optimize

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
 
def random_weights(num_assets):
    weights = np.random.rand(num_assets)
    return weights / sum(weights)

def stock_returns(data, log_returns=False, annual_days = 252):
    if log_returns:
       daily_returns = data.pct_change().apply(lambda x: np.log(1+x)) 
       returns = daily_returns.dropna(axis=0, how='all')
       covariance_matrix = returns.cov() * annual_days
    else:
       daily_returns = data.pct_change()
       returns = daily_returns.dropna(axis=0, how='all')
       covariance_matrix = returns.cov() * annual_days
    return (returns * annual_days).mean(), covariance_matrix

def portfolio_performance(weights, mean_returns, covariance_matrix):
    returns = mean_returns @ weights 
    portfolio_variance = weights.T @ covariance_matrix @ weights
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_std, returns

def neg_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate=0.02):
    p_var, p_ret = portfolio_performance(weights, mean_returns, covariance_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, covariance_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = optimize.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return [round(it, 3) for it in result.x]

def portfolio_volatility(weights, mean_returns, covariance_matrix):
    return portfolio_performance(weights, mean_returns, covariance_matrix)[0]

def min_volatility(mean_returns, covariance_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = optimize.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    # summarize the result
    #print('Status : %s' % result['message'])
    return [round(it, 3) for it in result.x]

def efficient_return(mean_returns, covariance_matrix, target_return=0.055):
    """
    Efficient return (minimize risk given a target return)
    """
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix)

    def portfolio_return(weights):
        return portfolio_performance(weights, mean_returns, covariance_matrix)[1]
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = optimize.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return [round(it, 3) for it in result.x]

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, covariance_matrix, ret))
    return efficients

def diversification_ratio(weights, mean_returns, covariance_matrix):
    """
    The weighted average of volatility divided by the portfolio volatility
    """
    portfolio_std, returns = portfolio_performance(weights, mean_returns, covariance_matrix)
    weights_std = np.dot(np.sqrt(np.diag(covariance_matrix)), weights.T)
    diversification_ratio = weights_std / portfolio_std 
    # return negative for minimization problem (maximize = minimize -)
    return -diversification_ratio


def max_diversification(mean_returns, covariance_matrix, long_only=True):
    """
    Maximum Diversification
    Reference Choueifaty, Yves, and Yves Coignard. 2008. “Toward Maximum Diversification.”
    paper https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf
    """
    # long only: long only constraint
    args = (mean_returns, covariance_matrix)
    num_assets = len(mean_returns)
    bounds = tuple((0,1) for asset in range(num_assets))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    if long_only: # add in long only constraint
        cons = ({'type': 'ineq', 'fun':  lambda x: np.sum(x) - 1})
    result = optimize.minimize(diversification_ratio, num_assets*[1./num_assets,],  args=args, bounds=bounds, method='SLSQP', constraints=cons)
    return [round(it, 3) for it in result.x]

def monte_carlo_optimisation(mean_returns, covariance_matrix, num_portfolios, risk_free_rate=0.02):
    """
    Monte Carlo Simulation
    """
    portfolio_returns = [] # Define an empty array for portfolio returns
    portfolio_volatility = [] # Define an empty array for portfolio volatility
    portfolio_weights = [] # Define an empty array for asset weights
    
    for portfolio in range(num_portfolios):
        weights = random_weights(len(mean_returns)) 
        volatility, returns = portfolio_performance(weights, mean_returns, covariance_matrix)
        portfolio_volatility.append(volatility)
        portfolio_weights.append(weights)
        portfolio_returns.append(returns)
            
    data = {'Returns': portfolio_returns, 'Volatility': portfolio_volatility}
    for counter, symbol in enumerate(mean_returns.index.tolist()):
        data[symbol+' weight'] = [w[counter] for w in portfolio_weights]
    # Dataframe of the 10000 portfolios created
    portfolios = pd.DataFrame(data)

    assert int(portfolios.iloc[0,2:].values.sum()) <= 1, "Weights sum should be equal 1"
    portfolios.head() 
    
    minimum_volatility_portfolio = portfolios.iloc[portfolios['Volatility'].idxmin()] # idxmin() gives us the minimum value in the column 'Volatility'.
    max_sharpe_ratio = portfolios.iloc[((portfolios['Returns']- risk_free_rate) / portfolios['Volatility']).idxmax()]
    
    return minimum_volatility_portfolio, max_sharpe_ratio
