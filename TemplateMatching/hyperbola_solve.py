import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import project_1D
from scipy.optimize import fsolve




def solve(CSV, IMG_TYPE, IMAGE_HEIGHT, filter):

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir,'processed', "filter data"))
    df = pd.read_csv(f"{CSV[:len(CSV)-4]}_{filter}.csv")
    os.chdir(current_dir)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    matrix_t = [x**2, x*y, y**2, x, y]
    matrix = np.transpose(matrix_t)
    solved = np.matmul(np.linalg.inv(np.matmul(matrix_t, matrix)),np.matmul(matrix_t, np.ones(np.shape(matrix)[0])))
    (A,B,C,D,E) = solved
    ends = np.roots([A, IMAGE_HEIGHT*B+D, -1+C*IMAGE_HEIGHT**2+E*IMAGE_HEIGHT])

    # fit = plot_hyperbola_linear(min(ends), max(ends), solved)
    try: 
        fit = equidistant_set(min(ends), max(ends), solved)
    except RuntimeError as err:
        raise RuntimeError(err)
    
    # img = cv2.imread(f'{csv[:len(csv)-4]}.jpg')
    img_path = os.path.join('img', f"{CSV[:len(CSV)-4]}{IMG_TYPE}")
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    fig2, intensity_ax = plt.subplots()
    ax.imshow(img, cmap=mpl.colormaps['gray'])
    
    projected_img = []

    ax.plot(fit[0], fit[1], '.-r', label="fit")
    target = os.path.join(current_dir,"processed", "fit visualization")
    os.chdir(target)
    fig.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)

    for i in range(len(fit[0])):
        projection = project_1D.project_one(fit[0][i], fit[1][i], solved)
        temp = []
        for j in range(len(projection[0])):
            try:
                pixel = img[projection[1][j],projection[0][j]]
                temp.append([pixel[0], pixel[1], pixel[2]])
            except IndexError:
                temp.append([255,255,255])
        projected_img.append(temp)
        ax.plot(projection[0], projection[1], '.-y', label="projection")

    avg_intensity = np.mean(projected_img, axis=1)
    avg_intensity_graph = [255-i[0] for i in avg_intensity]
    avg_intensity_map = []
    for _ in range(50):
        avg_intensity_map.append(avg_intensity)

    intensity_ax.bar(range(len(avg_intensity_graph)),avg_intensity_graph, color='k', align='center', width=1.0, label="fit")

    target = os.path.join(current_dir,"processed", "projection sampling")
    os.chdir(target)
    fig.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)

    target = os.path.join(current_dir,"processed", "projection graphed")
    os.chdir(target)
    fig2.savefig(f"{CSV[0:len(CSV)-4]}.jpg")
    os.chdir(current_dir)

    target = os.path.join(current_dir,"processed", "projection")
    os.chdir(target)
    cv2.imwrite(f"{CSV[0:len(CSV)-4]}.jpg", np.array(projected_img))
    os.chdir(current_dir)

    target = os.path.join(current_dir,"processed", "projection gradient")
    os.chdir(target)
    cv2.imwrite(f"{CSV[0:len(CSV)-4]}.jpg", np.array(avg_intensity_map))
    os.chdir(current_dir)

def equidistant_set(start, end, coeff):

    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1

    #Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    #circle: x = prev_x + cos(t); y = prev_y + sin(t)
    #intersect: Ax**2+Bxy+Cy**2+Dx+Ey-1=0
    (A,B,C,D,E) = coeff

    start_roots = [r for r in np.roots([quadratic[0], linear[0], constant[0]]) if r >= 0]
    start_y = min(start_roots)

    prev_x = start
    prev_y = start_y

    def func(t):
        return A*(prev_x+np.cos(t))**2+B*(prev_x+np.cos(t))*(prev_y+np.sin(t))+C*(prev_y+np.sin(t))**2+D*(prev_x+np.cos(t))+E*(prev_y+np.sin(t))-1


    result = ([],[])
    prev_roots = [0, 0]
    while prev_x < end:
        roots = fsolve(func, prev_roots)
        prev_roots = roots
        assert len(roots) == 2, "more than 2 roots found for unit circle"
        r1x, r1y = prev_x+np.cos(roots[0]), prev_y+np.sin(roots[0])
        r2x, r2y = prev_x+np.cos(roots[1]), prev_y+np.sin(roots[1])
        if r1x > r2x:
            result[0].append(r1x)
            result[1].append(r1y)
            prev_x = r1x
            prev_y = r1y
        else: 
            result[0].append(r2x)
            result[1].append(r2y)
            prev_x = r2x
            prev_y = r2y

        if len(result[0]) > 5000:
            raise RuntimeError("Hyperbola Fit Possibly Incorrect - Large Equidistance Arclength Set")
    
    return result
    

def plot_hyperbola_linear(start, end, coeff):
    # equidistant in x
    x = np.linspace(start, end, num=int(end-start)+1)
    quadratic = coeff[2]*np.ones(len(x))
    linear = coeff[1]*x+coeff[4]
    constant = coeff[0]*x**2+coeff[3]*x-1
    result = ([],[])
    for i in range(len(x)):
        roots = np.roots([quadratic[i], linear[i], constant[i]])
        for r in roots:
            if r >= 0:
                result[0].append(x[i])
                result[1].append(r)
    return result
