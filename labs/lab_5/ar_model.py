import numpy as np

def ar_series(burnin: int, n: int, c: float, o: np.array, seed: int = None) -> tuple[np.array, str]:
    """
    Generate an ARIMA series with the given parameters.

    Parameters
    ----------
    burnin : int
        Number of burn-in samples to generate before starting the series.
    n : int
        Number of samples to generate.
    c : float
        Constant term in the ARIMA model.
    o : np.array
        Coefficients of the autoregressive part of the ARIMA model.

    Returns
    -------
    np.array
        The generated ARIMA series.
    """
    if burnin < 0:
        raise ValueError("The burn-in period must be a non-negative integer.")
    if n <= 0:
        raise ValueError("The number of samples must be a positive integer.")
    if not isinstance(c, (int, float)):
        raise ValueError("The constant term must be a number.")
    if not isinstance(o, np.ndarray):
        raise ValueError("The moving average coefficients must be a numpy array.")
    if len(o) == 0:
        raise ValueError("The moving average coefficients must not be empty.")
    
    if seed is not None:
        np.random.seed(seed)

    e = np.random.randn(n + burnin)
    p = len(o)
    y = np.zeros(n + burnin)
    for t in range(n+burnin):
        y[t] = c + np.sum([o[i] * y[t-i-1] for i in range(p) if t-i-1 >= 0]) + e[t]

    formula = f"AR model: y[t] = {c} {' '.join([f'{o[i]:+}y[t-{i+1}]' for i in range(p)])} + e[t]"
    print(formula)
    return y[burnin:], formula