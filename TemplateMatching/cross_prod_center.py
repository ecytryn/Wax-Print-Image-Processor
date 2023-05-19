import numpy as np
import pandas as pd
import os
from utils import CONFIG, Cross

def crossProdCenter(coeff, df):
    axisSym = axisSymmetry(coeff)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    points = CONFIG.CROSS * 2 + 1

    if len(x) < points: 
        raise RuntimeError(f"Unable to perform {CONFIG.CROSS} crosses with {len(x)} points; {points} points needed")
    
    centerIndex = CONFIG.CROSS
    crossSums = []

    while centerIndex < len(x) - CONFIG.CROSS:
        crossSum = 0
        for c in range(CONFIG.CROSS):
            p1i  = centerIndex + c + 1
            p2i =  p1i * -1

            vec1 = (x[p1i], y[p1i])
            vec2 = (x[p2i], y[p2i])

            vecDiff = np.subtract(vec1, vec2)
            
            if CONFIG.CROSS_METHOD == Cross.SQAURED:
                crossSum += np.dot(vecDiff, axisSym)**2
            elif CONFIG.CROSS_METHOD == Cross.ABS:
                crossSum += abs(np.dot(vecDiff, axisSym))
        crossSums.append(crossSum)
        centerIndex += 1
    return np.argmin(crossSums) + CONFIG.CROSS


def axisSymmetry(coeff):
    (A,B,C,D,E) = coeff
    tanTwoTheta = B/(A-C)
    theta = np.arctan(tanTwoTheta)/2
    # symVector = (1, np.tan(theta)) # this one divides hyperbola into two, not bisect
    bisectVector = np.array([np.tan(theta), -1])
    return bisectVector