import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    xarr, yarr = np.asarray(x), np.asarray(y)
    return float(sum(np.abs(xarr - yarr)))