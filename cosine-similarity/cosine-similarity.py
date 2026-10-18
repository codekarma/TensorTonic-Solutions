import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    norm_prod = np.linalg.norm(a_arr, ord=2) * np.linalg.norm(b_arr, ord=2)
    if norm_prod == 0:
        return 0.0
    return float(a_arr @ b_arr / norm_prod)
    