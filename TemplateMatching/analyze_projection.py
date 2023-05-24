import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.signal import find_peaks
from utils import CONFIG

def avg_intensity(img_name: str, file_type: str, data):
    curr_dir = os.getcwd()
    target = os.path.join(curr_dir,"processed", "projection", img_name)
    os.chdir(target)
    fig2, ax2 = plt.subplots()

    fig2.set_figwidth(CONFIG.WIDTH_SIZE)
    fig2.set_figheight(CONFIG.HEIGHT_SIZE)
    fig2.tight_layout()

    avg_intensity = np.mean(data, axis=1)
    avg_window_intensity = []
    for i in range(len(avg_intensity)):
        if i - int(CONFIG.WINDOW_WIDTH/2) < 0:
            start = 0
        else: 
            start = i - int(CONFIG.WINDOW_WIDTH/2)
        if i + int(CONFIG.WINDOW_WIDTH/2) > len(data):
            end = len(data)
        else: 
            end = i + int(CONFIG.WINDOW_WIDTH/2)
        avg_window_intensity.append(np.mean(avg_intensity[start:end], axis=0))
    
    avg_intensity_graph = [255-i[0] for i in avg_window_intensity]

    # assumes image is at current dir
    projection = cv2.imread(f"projection{file_type}")

    ax2.imshow(projection)
    ax2.plot(range(len(avg_intensity_graph)),avg_intensity_graph, color='y')
    local_max_index, _ = find_peaks(avg_intensity_graph, distance=30)
    ax2.scatter(local_max_index,np.ones(len(local_max_index)), color='r')


    fig2.savefig(f"projection intensity{file_type}")
    os.chdir(curr_dir)

def arclength_histogram(img_name: str, file_type: str, arclength_location: list):
    curr_dir = os.getcwd()
    target = os.path.join(curr_dir,"processed", "projection", img_name)
    os.chdir(target)
    
    distances = []
    for location in range(len(arclength_location)-1):
        distances.append(arclength_location[location+1] - arclength_location[location])


    hist, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, tight_layout=True)
    ax1.hist(distances, bins=10)
    ax2.hist(distances, bins=20)
    ax3.hist(distances, bins=30)
    ax4.hist(distances, bins=40)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    hist.savefig(f"distance histogram{file_type}")
    os.chdir(curr_dir)

    suspect_distances = [ind for ind in range(len(distances)) 
                         if (distances[ind] <= CONFIG.ERROR_LOWER_B or distances[ind] >= CONFIG.ERROR_UPPER_B)]

    res = []
    for dist in suspect_distances:
        res.append(dist)
        res.append(dist + 1)
    
    return res