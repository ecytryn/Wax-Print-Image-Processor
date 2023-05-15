import os
import pandas as pd
import matplotlib.pyplot as plt


from utils import suffix, CONFIG


def plot_manual():
    data = sorted([file for file in os.listdir(CONFIG.PATH) if suffix(file) == ".csv"])
    tooth = []
    tooth_y = []
    gap = []
    gap_y = []
    center_t = []
    center_t_y = []
    center_g = []
    center_g_y = []

    for i in range(len(data)):
        
        file_path = os.path.join(CONFIG.PATH, data[i])
        if not os.path.isfile(file_path):
            raise RuntimeError(f"data for {data[i]} cannot be found in {CONFIG.PATH}; did you try manual?")
        df = pd.read_csv(file_path)
        t = i * CONFIG.TIME

        center_tooth = df[df["type"] == "Tooth.CENTER_T"]["x"].to_numpy()
        if len(center_tooth) == 0:
            center = df[df["type"] == "Tooth.CENTER_G"]["x"].to_numpy()[0]
            center_g.append(0)
            center_g_y.append(t)
        else: 
            center = center_tooth[0]
            center_t.append(0)
            center_t_y.append(t)
        
        new_tooth = list(df[df["type"] == "Tooth.TOOTH"]["x"].to_numpy()-center)
        tooth_y += [t for _ in range(len(new_tooth))]
        tooth += new_tooth

        new_gap = list(df[df["type"] == "Tooth.GAP"]["x"].to_numpy()-center)
        gap_y += [t for _ in range(len(new_gap))]
        gap += new_gap

    fig, ax = plt.subplots()
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    ax.scatter(tooth, tooth_y, c="c")
    ax.scatter(gap, gap_y, c="#c8c8c8")
    ax.scatter(center_g, center_g_y, c="r")
    ax.scatter(center_t, center_t_y, c="r")

    fig.savefig("res.png")

def plot_manual_even():
    data = sorted([file for file in os.listdir(CONFIG.PATH) if suffix(file) == ".csv"])
    tooth = []
    tooth_y = []
    gap = []
    gap_y = []
    center_t = []
    center_t_y = []
    center_g = []
    center_g_y = []

    for i in range(len(data)):

        file_path = os.path.join(CONFIG.PATH, data[i])
        if not os.path.isfile(file_path):
            raise RuntimeError(f"data for {data[i]} cannot be found in {CONFIG.PATH}; did you try manual?")
        df = pd.read_csv(file_path)
        t = i * CONFIG.TIME

        center_tooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        if len(center_tooth) == 0:
            center = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
            center_g.append(0)
            center_g_y.append(t)
        else: 
            center = center_tooth[0]
            center_t.append(0)
            center_t_y.append(t)
        
        new_tooth = list(df.index[df["type"] == "Tooth.TOOTH"].to_numpy()-center)
        tooth_y += [t for _ in range(len(new_tooth))]
        tooth += new_tooth

        new_gap = list(df.index[df["type"] == "Tooth.GAP"].to_numpy()-center)
        gap_y += [t for _ in range(len(new_gap))]
        gap += new_gap

    fig, ax = plt.subplots()
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    ax.scatter(tooth, tooth_y, c="c")
    ax.scatter(gap, gap_y, c="#c8c8c8")
    ax.scatter(center_g, center_g_y, c="r")
    ax.scatter(center_t, center_t_y, c="r")

    fig.savefig("res_even.png")