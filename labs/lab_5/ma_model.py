import numpy as np

def ma_series(burnin: int, n: int, c: float, O: np.array, seed: int=None) -> tuple[np.array, str]:
    """
    Generate a moving average series.

    Parameters
    ----------
    burnin : int
        The number of burn-in samples to generate.
    n : int
        The number of samples to generate.
    c : float
        The constant term in the moving average model.
    O : np.array
        The moving average coefficients.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    np.array
        The generated moving average series.

    Raises
    ------
    ValueError
        If the burn-in period is negative.
        If the number of samples is non-positive.
        If the constant term is not a number.
        If the moving average coefficients are not a numpy array.
        If the moving average coefficients are empty.
    """
    if burnin < 0:
        raise ValueError("The burn-in period must be a non-negative integer.")
    if n <= 0:
        raise ValueError("The number of samples must be a positive integer.")
    if not isinstance(c, (int, float)):
        raise ValueError("The constant term must be a number.")
    if not isinstance(O, np.ndarray):
        raise ValueError("The moving average coefficients must be a numpy array.")
    if len(O) == 0:
        raise ValueError("The moving average coefficients must not be empty.")
    
    if seed is not None:
        np.random.seed(seed)
    e = np.random.normal(size=n+burnin)
    # append 1 to the beggining of O
    q = len(O)
    y = np.zeros(n + burnin)
    for t in range(n+burnin):
        y[t] = c + e[t] + np.sum([O[i] * e[t-i-1] for i in range(q) if t-i-1 >= 0])

    formula = f"MA model: y[t] = {c} +e[t] {' '.join([f'{O[i]:+}e[t-{i+1}]' for i in range(q)])}"
    print(formula)
    return y[burnin:], formula