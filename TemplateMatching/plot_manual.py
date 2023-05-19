import os
import pandas as pd
import matplotlib.pyplot as plt


from utils import suffix, CONFIG


def plotManual():
    data = sorted([file for file in os.listdir(CONFIG.PATH) if suffix(file) == ".csv"])
    tooth = []
    toothY = []
    gap = []
    gapY = []
    centerT = []
    centerTY = []
    centerG = []
    centerGY = []

    for i in range(len(data)):
        
        filePath = os.path.join(CONFIG.PATH, data[i])
        if not os.path.isfile(filePath):
            raise RuntimeError(f"data for {data[i]} cannot be found in {CONFIG.PATH}; did you try manual?")
        df = pd.read_csv(filePath)
        t = i * CONFIG.TIME

        centerTooth = df[df["type"] == "Tooth.CENTER_T"]["x"].to_numpy()
        if len(centerTooth) == 0:
            center = df[df["type"] == "Tooth.CENTER_G"]["x"].to_numpy()[0]
            centerG.append(0)
            centerGY.append(t)
        else: 
            center = centerTooth[0]
            centerT.append(0)
            centerTY.append(t)
        
        newTooth = list(df[df["type"] == "Tooth.TOOTH"]["x"].to_numpy()-center)
        toothY += [t for _ in range(len(newTooth))]
        tooth += newTooth

        newGap = list(df[df["type"] == "Tooth.GAP"]["x"].to_numpy()-center)
        gapY += [t for _ in range(len(newGap))]
        gap += newGap

    fig, ax = plt.subplots()
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    ax.scatter(tooth, toothY, c="c")
    ax.scatter(gap, gapY, c="#c8c8c8")
    ax.scatter(centerG, centerGY, c="r")
    ax.scatter(centerT, centerTY, c="r")

    fig.savefig("res.png")

def plotManualEven():
    data = sorted([file for file in os.listdir(CONFIG.PATH) if suffix(file) == ".csv"])
    tooth = []
    toothY = []
    gap = []
    gapY = []
    centerT = []
    centerTY = []
    centerG = []
    centerGY = []

    for i in range(len(data)):

        filePath = os.path.join(CONFIG.PATH, data[i])
        if not os.path.isfile(filePath):
            raise RuntimeError(f"data for {data[i]} cannot be found in {CONFIG.PATH}; did you try manual?")
        df = pd.read_csv(filePath)
        t = i * CONFIG.TIME

        centerTooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        if len(centerTooth) == 0:
            center = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
            centerG.append(0)
            centerGY.append(t)
        else: 
            center = centerTooth[0]
            centerT.append(0)
            centerTY.append(t)
        
        newTooth = list(df.index[df["type"] == "Tooth.TOOTH"].to_numpy()-center)
        toothY += [t for _ in range(len(newTooth))]
        tooth += newTooth

        newGap = list(df.index[df["type"] == "Tooth.GAP"].to_numpy()-center)
        gapY += [t for _ in range(len(newGap))]
        gap += newGap

    fig, ax = plt.subplots()
    fig.set_figwidth(CONFIG.WIDTH_SIZE)
    fig.set_figheight(CONFIG.HEIGHT_SIZE)
    ax.scatter(tooth, toothY, c="c")
    ax.scatter(gap, gapY, c="#c8c8c8")
    ax.scatter(centerG, centerGY, c="r")
    ax.scatter(centerT, centerTY, c="r")

    fig.savefig("res_even.png")