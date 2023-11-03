import pandas as pd
import numpy as np
from scipy import optimize as op
import plotly.express as px


def portfolio(weights, expected_return, cov_matrix, period=252):
    """
    This function handles the calculations for modern portfolio theory. All inputs should have the same shapes/number of assets.

    - weights         : numpy.array : the proportion of each asset to include in the portfolio.
    - expected_return : numpy.array : of the expected return of assets.
    - cov_matrix      : numpy.array : covariance matrix of the assets used.

    portfolio -> tuple(portfolio return, variance)
    """
    # We calcuate the returns of the portfolio.
    # Multiply by the holding period, 252 for the amount of trading days in a year.
    weighted_return = expected_return @ weights * period
    # Variance or risk of the portfolio is calculated using the covariance table and weights.
    # Multiplying by the square root of the holding period scales the variance to the holding period.
    # The square root of time rule is also found in the black scholes equation.
    variance = (weights @ (cov_matrix @ weights.T)) * np.sqrt(period)

    return weighted_return, variance


def constraint_fun(weights):
    """
    This is an equality constraint for the weights.

    The weights have to add to one, but an equality constraint is only valid when it is equal to zero.

    sum(weights) = 1

    sum(weights) - 1 = 0
    """
    return sum(weights) - 1


def find_low_risk(data):
    """
    This function takes the data and outputs the returns and variance of the lowest risk portfolio.

    - data : pandas.DataFrame

    find_low_risk -> dict(return : float, variance : float, weights : dict(asset : float))
    """

    # For scipy it needs initial values to make it's guess, it doesn't even have to match the bounds or contraints.
    x_0 = np.array([0.2] * len(data.columns))
    # Constraint is a dictionary which defines the type and what the function to use is.
    constraint = {"type": "eq", "fun": constraint_fun}
    # Bounds keeps the variables in a certain space, in this case it is positive values.
    # Without bounds its possible to find portfolios with negative weights..
    bounds = op.Bounds(0, 1)

    # This function specifies what to minimise, it is essentially a wrapper for the function portfolio.
    def target_low_risk(weights):
        return portfolio(weights, returns, cov_matrix)[1]
    
    # This is the data we need for the portfolio function defined earlier.
    returns = data.mean() # Our expected returns.
    cov_matrix = data.cov() # Covariance table.

    # The minimise function takes all the inputs and finds the lowest value for the target function.
    # minimise returns a OptimizeResult object which contains information about the results.
    low_risk = op.minimize(target_low_risk, x_0, constraints=constraint, bounds=bounds)
    # OptimizeResult.x gives the input variables and OptimizeResult.fun gives the output of the target function at those variables.
    # OptimizeResult also has information about the optimisation such as if it was a success.
    
    # To get the optimised returns and variance we can put in the same data and the variables optimise found.
    returns, variance = portfolio(low_risk.x, returns, cov_matrix)
    # Output the weights of each asset.
    weights_output = dict(zip(data.columns, low_risk.x))

    return {"return" : returns, "variance" : variance, "weights" : weights_output}

# To find an efficient portfolio we will need a different target function.
def find_sharpe(data, risk_free=0.05):
    """
    Maximises the sharpe ratio of a portfolio.

    - data : pandas.DataFrame

    find_sharpe -> dict(return : float, variance : float, weights : dict(asset : float))
    """

    x_0 = np.array([0.2] * len(data.columns))
    contraint = {"type": "eq", "fun": constraint_fun}
    bounds = op.Bounds(0, 1)

    exp_returns = data.mean()
    cov_matrix = data.cov()

    # To finds an efficient portfolio, target a high sharpe ratio.
    def target_sharpe(weights):
        returns, variance = portfolio(weights, exp_returns, cov_matrix)
        # Calcuate sharpe.
        sharpe = (returns - risk_free) / variance
        # Minimise finds the lowest possible value so we need to make sharpe negative so that the absolute value is maximised.
        return -sharpe

    # The rest of the function is the same as before.
    sharpe = op.minimize(target_sharpe, x_0, constraints=contraint, bounds=bounds)
    
    returns, variance = portfolio(sharpe.x, exp_returns, cov_matrix)
    weights_output = dict(zip(data.columns, sharpe.x))

    return {"return" : returns, "variance" : variance, "weights" : weights_output}

# This is where the code runs.
# The if statement means it wont run if you import it.
if __name__ == "__main__":

    # Get and process data.
    data = pd.read_excel("data.xlsx", index_col=0)
    data = data.pct_change()
    data = data.dropna()

    # Call the functions.
    sharpe = find_sharpe(data)
    low_risk = find_low_risk(data)

    # print("sharpe : ")
    # print(sharpe)
    # print("------")
    # print("low risk : ")
    # print(low_risk)

    # The output dictionary adds the stocks and found portfolios to a dictionary to be turned into a pandas dataframe.
    output_dict = {"return" : list(data.mean() * 252) + [sharpe["return"], low_risk["return"]], "variance" : list((data.std() ** 2) * np.sqrt(252)) + [sharpe["variance"], low_risk["variance"]]}
    frame = pd.DataFrame(output_dict)

    # Plot all the points using a graphing libary.
    px.scatter(frame, x="variance", y="return").show()