import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    return float(np.sqrt(np.sum((x_arr - y_arr) * (x_arr - y_arr))))