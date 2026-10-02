import numpy as np


def calculate_covariance_matrix(sigma, correlation):
    """
    Calculate the covariance matrix from volatilities
    and a correlation matrix.
    """
    return np.outer(sigma, sigma) * correlation


def minimum_variance_weights(covariance):
    """
    Calculate minimum-variance weights for a two-asset portfolio.
    """
    sigma_1_sq = covariance[0, 0]
    sigma_2_sq = covariance[1, 1]
    sigma_12 = covariance[0, 1]

    weight_1 = (
        sigma_2_sq - sigma_12
    ) / (
        sigma_1_sq + sigma_2_sq - 2 * sigma_12
    )

    weight_2 = 1 - weight_1

    return np.array([weight_1, weight_2])


def portfolio_volatility(weights, covariance):
    """
    Calculate portfolio volatility from weights
    and a covariance matrix.
    """
    variance = weights @ covariance @ weights

    return np.sqrt(variance)


def calculate_var_es(
    portfolio_values,
    confidence_level=0.95
):
    """
    Calculate Value at Risk and Expected Shortfall.
    """
    percentile = (1 - confidence_level) * 100

    var_price = np.percentile(
        portfolio_values,
        percentile
    )

    initial_value = portfolio_values.mean()

    var = initial_value - var_price

    worst_values = portfolio_values[
        portfolio_values <= var_price
    ]

    expected_shortfall = (
        initial_value - worst_values.mean()
    )

    return var, expected_shortfall