import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from utils import CONFIG


def project_one(x, y, coeff):
    (A, B, C, D, E) = coeff
    
    discrim = (B*x+E)**2-4*C*(A*x**2+D*x-1)
    tangent = (1, 1/(2*C)*(-B+((2*B*(B*x+E))-4*C*(2*A*x+D))/(2*np.sqrt(discrim))))
    normal = (1, -1/(tangent[1]))
    normal_h = (normal[0]/np.sqrt((normal[0])**2+(normal[1])**2), normal[1]/np.sqrt((normal[0])**2+(normal[1])**2))

    normal_x = []
    normal_y = []

    c = CONFIG.SAMPLING_WIDTH
    while c >= -CONFIG.SAMPLING_WIDTH:
        normal_x.append(int(x+c*normal_h[0]))
        normal_y.append(int(y+c*normal_h[1]))
        c -= 1
    
    df = pd.DataFrame({'x': normal_x, 'y': normal_y})
    df.sort_values(by=['y'], inplace=True)

    return (df['x'].to_numpy(), df['y'].to_numpy(), normal_h, tangent)


def proj_data(x, y, coeff):
    (A,B,C,D,E) = coeff
    solved = fsolve(func, x, [x, y, coeff])
    hyperbola_x = solved[0]
    assert len(solved) == 1, f"More than one solution found for closest point to the hyperbola form {x,y}, {solved}, {coeff}"
    discrim = (B*hyperbola_x+E)**2-4*C*(A*hyperbola_x**2+D*hyperbola_x-1)
    hyperbola_y = ((-B*hyperbola_x-E+np.sqrt(discrim))/(2*C))
    distance = np.sqrt((x-hyperbola_x)**2+(y-hyperbola_y)**2)

    if y >= hyperbola_y: # if the data point is on the inside of the jaw
        return (hyperbola_x, distance)
    else:
        return (hyperbola_x, -distance)


def func(t, args):
    (x, y, coeff) = args
    (A,B,C,D,E) = coeff
    discrim = (B*t+E)**2-4*C*(A*t**2+D*t-1)
    tangent = (1, 1/(2*C)*(-B+((2*B*(B*t+E))-4*C*(2*A*t+D))/(2*np.sqrt(discrim))))
    normal = (t-x, ((-B*t-E+np.sqrt(discrim))/(2*C))-y)
    return tangent[0]*normal[0]+tangent[1]*normal[1]
