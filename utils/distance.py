import numpy as np


def try_divide(x, y, val=0.0):
    """
    try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def cosine_distance(v1, v2):
    """
    余弦距离
    return cos score
    """
    up = float(np.sum(v1 * v2))
    down = np.linalg.norm(v1) * np.linalg.norm(v2)
    return try_divide(up, down)