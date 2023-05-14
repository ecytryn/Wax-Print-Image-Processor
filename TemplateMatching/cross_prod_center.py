import numpy as np
import pandas as pd
import os
from utils import CONFIG, Cross

def cross_prod_center(coeff, df):
    axis_sym = axis_symmetry(coeff)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    points = CONFIG.CROSS * 2 + 1

    if len(x) < points: 
        raise RuntimeError(f"Unable to perform {CONFIG.CROSS} crosses with {len(x)} points; {points} points needed")
    
    center_index = CONFIG.CROSS
    cross_sums = []

    while center_index < len(x) - CONFIG.CROSS:
        cross_sum = 0
        for c in range(CONFIG.CROSS):
            p1_i  = center_index + c + 1
            p2_i =  p1_i * -1

            vec_1 = (x[p1_i], y[p1_i])
            vec_2 = (x[p2_i], y[p2_i])

            vec_dif = np.subtract(vec_1, vec_2)
            
            if CONFIG.CROSS_METHOD == Cross.SQAURED:
                cross_sum += np.dot(vec_dif, axis_sym)**2
            elif CONFIG.CROSS_METHOD == Cross.ABS:
                cross_sum += abs(np.dot(vec_dif, axis_sym))
        cross_sums.append(cross_sum)
        center_index += 1
    return np.argmin(cross_sums) + CONFIG.CROSS


def axis_symmetry(coeff):
    (A,B,C,D,E) = coeff
    tan_twoheta = B/(A-C)
    theta = np.arctan(tan_twoheta)/2
    # sym_vector = (1, np.tan(theta)) # this one divides hyperbola into two, not bisect
    bisect_vector = np.array([np.tan(theta), -1])
    return bisect_vector