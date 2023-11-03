import pandas as pd
import numpy as np
from scipy import optimize as op
import plotly.express as px

# Classes containerize data, variables and functions which keeps everything organised . This is also known as Object orientated programming
# They also provide far greater flexibility for being able to initialise values and overwrite normal methods such as addition
# This is not a tutorial for classes but this is an example of how powerful classes can be.
class Portfolio_Theory:
    """
    Contains the data and functions for modern portfolio theory.
    """
    # Class variables
    risk_free_rate = 0.05
    period = 252

    def __init__(self, data) -> None:
        # Instance variables
        self.data = data
        self.returns = data.mean()
        self.cov_matrix = data.cov()

    def portfolio(self, weights, expected_return, cov_matrix):
        """
        Calculates return and risk of portfolio for the weights.
        """
        weighted_return = (expected_return @ weights) * Portfolio_Theory.period
        variance = (weights @ (cov_matrix @ weights.T)) * np.sqrt(Portfolio_Theory.period)

        return weighted_return, variance

    def find(self, target):
        """
        Finds variables based on the target function.
        """
        x_0 = np.array([0.2] * len(self.data.columns))
        constraint = {"type": "eq", "fun": lambda x: sum(x) - 1}

        result = op.minimize(target, x_0, constraints=constraint, bounds=op.Bounds(0, 1))

        returns, variance = self.portfolio(result.x, self.returns, self.cov_matrix)
        weights_output = dict(zip(self.data.columns, result.x))

        return {"return": returns, "variance": variance, "weights": weights_output}


    @property # This is a decorator. They are used to add to or change functions.
    def assets(self):
        data_for_frame = {"return": (data.mean() * Portfolio_Theory.period),
               "variance": ((data.std() ** 2) * np.sqrt(Portfolio_Theory.period))
               }
        return pd.DataFrame(data_for_frame, index=self.data.columns)


    @property
    def low_risk(self):

        def target_low_risk(weights):
            return self.portfolio(weights, self.returns, self.cov_matrix)[1]

        return self.find(target_low_risk)


    @property
    def sharpe(self):

        def target_sharpe(weights):
            returns, variance = self.portfolio(weights, self.returns, self.cov_matrix)
            sharpe = (returns - Portfolio_Theory.risk_free_rate) / np.sqrt(variance)
            return -sharpe

        return self.find(target_sharpe)

    def plot(self):
        px.scatter(self.assets, x="variance", y="return").show()


if __name__ == "__main__":
    # Data
    data = pd.read_excel("data.xlsx", index_col=0)
    data = data.pct_change()
    data = data.dropna()

    # An advantage of classes is that we can have mutiple instances of the same functions, but with different data at the same time.
    # Another advantage is that the end user needs very little code since the class is able to handle the rest.
    # Try import your own data into a new Portfolio_Theory class.
    Portfolio_Theory.risk_free_rate = 0.06
    mpt = Portfolio_Theory(data)
    print(mpt.low_risk)
    print(mpt.sharpe)
    print(mpt.assets)
    # modern_portfolio_theory(data).plot()
