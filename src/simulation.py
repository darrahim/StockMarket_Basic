import numpy as np


def simulate_gbm(
    n_years,
    n_scenarios,
    mu,
    sigma,
    steps_per_year,
    s_0
):
    """
    Simulate geometric Brownian motion for a single asset.
    """
    dt = 1 / steps_per_year
    n_steps = n_years * steps_per_year

    Z = np.random.normal(
        size=(n_steps, n_scenarios)
    )

    growth = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    prices = s_0 * np.exp(
        np.cumsum(growth, axis=0)
    )

    prices = np.vstack([
        np.full((1, n_scenarios), s_0),
        prices
    ])

    return prices


def simulate_correlated_gbm(
    n_years,
    n_scenarios,
    mu,
    sigma,
    correlation,
    steps_per_year,
    s_0
):
    """
    Simulate correlated geometric Brownian motion
    for multiple assets.
    """
    n_assets = len(mu)

    dt = 1 / steps_per_year
    n_steps = n_years * steps_per_year

    L = np.linalg.cholesky(correlation)

    Z = np.random.normal(
        size=(n_assets, n_steps, n_scenarios)
    )

    correlated_Z = L @ Z.reshape(
        n_assets, -1
    )

    correlated_Z = correlated_Z.reshape(
        n_assets,
        n_steps,
        n_scenarios
    )

    growth = (
        (
            mu[:, None, None]
            - 0.5 * sigma[:, None, None]**2
        ) * dt
        + sigma[:, None, None]
        * np.sqrt(dt)
        * correlated_Z
    )

    prices = s_0[:, None, None] * np.exp(
        np.cumsum(growth, axis=1)
    )

    initial_prices = np.broadcast_to(
        s_0[:, None, None],
        (n_assets, 1, n_scenarios)
    )

    prices = np.concatenate(
        [initial_prices, prices],
        axis=1
    )

    return prices