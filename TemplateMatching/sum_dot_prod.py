import numpy as np
import pandas as pd
import os
from utils import CONFIG, Cross

def sum_dot_prod(coeff, df):
    axis_sym = axis_symmetry(coeff)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    points = CONFIG.CROSS * 2 + 1

    if len(x) < points: 
        raise RuntimeError(f"Unable to perform {CONFIG.CROSS} crosses with {len(x)} points; {points} points needed")
    
    center_index = CONFIG.CROSS
    dot_sums = []

    while center_index < len(x) - CONFIG.CROSS:
        dot_sum = 0
        for c in range(CONFIG.CROSS):
            p1i  = center_index + c + 1
            p2i =  center_index - c - 1

            vec1 = (x[p1i], y[p1i])
            vec2 = (x[p2i], y[p2i])

            vec_diff = np.subtract(vec1, vec2)            
            if CONFIG.CROSS_METHOD == Cross.SQAURED:
                dot_sum += np.dot(vec_diff, axis_sym)**2
            elif CONFIG.CROSS_METHOD == Cross.ABS:
                dot_sum += abs(np.dot(vec_diff, axis_sym))
        dot_sums.append(dot_sum)
        center_index += 1
    

    return np.argmin(dot_sums) + CONFIG.CROSS


def axis_symmetry(coeff):
    (A,B,C,D,E) = coeff
    tan_twotheta = B/(A-C)
    theta = np.arctan(tan_twotheta)/2
    # sym_vector = (1, np.tan(theta)) # this one divides hyperbola into two, not bisect
    bisect_vector = np.array([np.tan(theta), -1])
    return bisect_vector