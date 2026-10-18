import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    adotb = np.dot(a_arr, b_arr)
    a_norm, b_norm = np.linalg.norm(a_arr, ord=2), np.linalg.norm(b_arr, ord=2)
    if not a_norm or not b_norm:
        return float(0)
    return float(adotb / (a_norm * b_norm))
    