import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import suffix, CONFIG, parseDate


def dataToCSV():

    data = folder_search() #sorted
    maxCenterIndex = 0
    maxLength = 0
    centerIndecies = []
    dates = []

    for i in range(len(data)):
        # parse date
        df = pd.read_csv(data[i])
        subdirname = os.path.basename(os.path.dirname(data[i]))
        date = parseDate(subdirname)
        dates.append(date)

        # find center index
        centerTooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
        if len(centerTooth) == 0:
            centerIndex = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
        else: 
            centerIndex = centerTooth[0]
        centerIndecies.append(centerIndex)
        
        # find max length, find max center index
        if centerIndex > maxCenterIndex:
            maxCenterIndex = centerIndex
        if len(df) > maxLength:
            maxLength = len(df)

    # set dataframe shape (1 column for date, the rest for teeth indecies)
    columns = range(maxCenterIndex + maxLength - 1) - maxCenterIndex
    columns = [str(column) for column in columns]
    dfOutputArclength = pd.DataFrame(columns=["date"]+columns)
    dfOutputBinary = pd.DataFrame(columns=["date"]+columns)

    entrySoFar = 0
    for i in range(len(dates)):
        # read data
        df = pd.read_csv(data[i])
        x = df["x"].to_numpy()
        types = df["type"]

        #arclength rep = x values of projection; binary data rep = 1 for teeth, 0 for gap
        arclengthDataRep = x
        binaryDataRep = [1 if (types[i] == "Tooth.TOOTH" or types[i] == "Tooth.CENTER_T") else 0 for i in range(len(x))]

        # add front and back padding to obtain the correct shape to insert into dataframe
        arclengthDataRepPad = padding(arclengthDataRep, centerIndecies[i], maxCenterIndex, len(columns))
        binaryDataRepPad = padding(binaryDataRep, centerIndecies[i], maxCenterIndex, len(columns))

        # prepend date for the "date" column
        dfEntryArclength = [dates[i]] + arclengthDataRepPad
        dfEntryBinary = [dates[i]] + binaryDataRepPad

        # set the maxlength'th entry to be the new entry; aka add new entry
        dfOutputArclength.loc[entrySoFar] = dfEntryArclength
        dfOutputBinary.loc[entrySoFar] = dfEntryBinary
        entrySoFar += 1

    # save to output folder
    dfOutputBinary.to_csv(os.path.join("processed", "output", "binary data.csv"))
    dfOutputArclength.to_csv(os.path.join("processed", "output", "arclength data.csv"))


def folder_search():
    '''
    return the list of csv within the "results" folder specified in CONFIG. 
    '''
    root = CONFIG.PATH
    allDir = [x[0] for x in os.walk(root)]
    csv = []
    
    for dir in allDir:
        items = os.listdir(dir)
        for item in items:
            if item.endswith("1D.csv"):
                csv.append(os.path.join(dir, item))
    
    return sorted(csv)


def padding(dataList, currentCenterIndex, targetCenterIndex, numOfColumns):
    '''
    Give dataList front padding to center its currentCenterIndex with the targetCenterIndex
    and give dateList back padding to have the same number of items as numOfColumns
    '''
    frontPad = targetCenterIndex - currentCenterIndex
    frontPadd = [None for _ in range(frontPad)]
    frontPadded = frontPadd+list(dataList)
    backPad = numOfColumns - len(frontPadded)
    backPadd = [None for _ in range(backPad)]
    return frontPadded+backPadd


def plotResult():
    pass
    # data = folder_search()

    # tooth = []
    # toothY = []
    # gap = []
    # gapY = []
    # centerT = []
    # centerTY = []
    # centerG = []
    # centerGY = []

    # for i in range(len(data)):
    #     df = pd.read_csv(data[i])
    #     subdirname = os.path.basename(os.path.dirname(data[i]))
    #     date = parseDate(subdirname)

    #     centerTooth = df.index[df["type"] == "Tooth.CENTER_T"].to_numpy()
    #     if len(centerTooth) == 0:
    #         centerIndex = df.index[df["type"] == "Tooth.CENTER_G"].to_numpy()[0]
    #         centerG.append(0)
    #         centerGY.append(date)
    #     else: 
    #         centerIndex = centerTooth[0]
    #         centerT.append(0)
    #         centerTY.append(date)

    #     newTeeth = list(df.index[df["type"] == "Tooth.TOOTH"].to_numpy()-centerIndex)
    #     toothY += [date for _ in range(len(newTeeth))]
    #     tooth += newTeeth

    #     newGap = list(df.index[df["type"] == "Tooth.GAP"].to_numpy()-centerIndex)
    #     gapY += [date for _ in range(len(newGap))]
    #     gap += newGap

    # fig, ax = plt.subplots()
    # fig.set_figwidth(CONFIG.WIDTH_SIZE)
    # fig.set_figheight(CONFIG.HEIGHT_SIZE)
    # ax.scatter(tooth, toothY, c="c")
    # ax.scatter(gap, gapY, c="#c8c8c8")
    # ax.scatter(centerG, centerGY, c="r")
    # ax.scatter(centerT, centerTY, c="r")

    # fig.savefig("res.png")