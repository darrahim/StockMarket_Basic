import statsmodels.api as sm


def run_factor_regression(asset_returns, factor_returns):
    """
    Run an OLS factor regression.

    Parameters
    ----------
    asset_returns : pandas Series
        Returns of the asset being explained.

    factor_returns : pandas DataFrame
        Returns of the explanatory factors.

    Returns
    -------
    statsmodels regression results
    """

    X = sm.add_constant(factor_returns)

    model = sm.OLS(
        asset_returns,
        X
    ).fit()

    return model