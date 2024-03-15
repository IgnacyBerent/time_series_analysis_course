import numpy as np

def exponential_smoothing(series: np.array, alpha: float,l0: float) -> float:
    if len(series) > 0:
        return alpha * series[-1] + (1-alpha) * exponential_smoothing(series[:-1], alpha, l0)
    else:
        return l0
    
def exponential_smoothing_sse(series: np.array, alpha: float, l0: float) -> float:
    T = len(series)
    sse = 0
    for t in range(1,T):
        yt = series[t]
        pred = exponential_smoothing(series[:t], alpha, l0)
        sse += (yt - pred)**2
    return sse

def find_best_alpha(series: np.array, l0: float) -> float:
    alphas = np.arange(0, 1, 0.01)
    best_alpha = alphas[0]
    best_sse = exponential_smoothing_sse(series, alphas[0], l0)
    for alpha in alphas[1:]:
        sse = exponential_smoothing_sse(series, alpha, l0)
        if sse < best_sse:
            best_sse = sse
            best_alpha = alpha
    return best_alpha