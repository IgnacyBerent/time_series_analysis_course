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
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple[np.array, str]
        The generated ARIMA series and the formula used to generate it.
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
    return y[burnin:], formula

def ar_series_linear(burnin: int, n: int, c: float, o: np.array, seed: int = None) -> tuple[np.array, str]:
    """
    Generate an ARIMA series with the given parameters using linear algebra approach.

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
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple[np.array, str]
        The generated ARIMA series and the formula used to generate it.
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
    y = [0]*p
    for t in range(p, n+burnin):
        y.append(c + np.dot(o, y[t-p:t]) + e[t])
        

    formula = f"AR model: y[t] = {c} {' '.join([f'{o[i]:+}y[t-{i+1}]' for i in range(p)])} + e[t]"
    return y[burnin:], formula