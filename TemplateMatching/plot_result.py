import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from utils import CONFIG, parseDate


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
        
        # update max length, update max center index
        if centerIndex > maxCenterIndex:
            maxCenterIndex = centerIndex
        if len(df) > maxLength:
            maxLength = len(df)

    # set dataframe shape (1 column for date + the rest for teeth indecies)
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
        
        #arclength representation = x values of projection; binary data representation = 1 for teeth, 0 for gap
        arclengthDataRep = x - df["x"][centerIndecies[i]]
        binaryDataRep = [1 if (types[i] == "Tooth.TOOTH" or types[i] == "Tooth.CENTER_T") else 0 for i in range(len(x))]

        # add front and back padding to obtain the correct shape to insert into dataframe
        arclengthDataRepPad = padding(arclengthDataRep, centerIndecies[i], maxCenterIndex, len(columns))
        binaryDataRepPad = padding(binaryDataRep, centerIndecies[i], maxCenterIndex, len(columns))

        # prepend date into the "date" column
        dfEntryArclength = [dates[i]] + arclengthDataRepPad
        dfEntryBinary = [dates[i]] + binaryDataRepPad

        # set the maxlength'th entry to be the new entry; aka add new entry
        dfOutputArclength.loc[entrySoFar] = dfEntryArclength
        dfOutputBinary.loc[entrySoFar] = dfEntryBinary
        entrySoFar += 1

    # save to output folder
    dfOutputBinary.to_csv(os.path.join("processed", "output", "binary data.csv"))
    dfOutputArclength.to_csv(os.path.join("processed", "output", "arclength data.csv"))

    plotResult()


def plotResult():
    """plot a representation of the output csv"""

    outputPath = os.path.join("processed", "output")
    dfArclength = pd.read_csv(os.path.join(outputPath, "arclength data.csv"))
    dfBinary = pd.read_csv(os.path.join(outputPath, "binary data.csv"))

    arcTooth = []
    arcGap = []
    arcCenterT = []
    arcCenterG = []

    binTooth = []
    binGap = []
    binCenterT = []
    binCenterG = []

    toothY = []
    gapY = []
    centerTY = []
    centerGY = []

    # convert string into dates
    dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dfArclength["date"]])
    # first two columns are: 1) index of df, 2) date
    indexColumns = dfArclength.columns.to_list()[2:] 

    for entryIndex in range(len(dates)):
        for columnIndex in indexColumns:
            arcEntry = dfArclength[columnIndex][entryIndex]
            binEntry = dfBinary[columnIndex][entryIndex]

            # if not empty
            if not pd.isna(binEntry):
                # if a tooth
                if int(binEntry) == 1:
                    # if column is the centerindex
                    if int(columnIndex) == 0:
                        arcCenterT.append(float(arcEntry))
                        binCenterT.append(float(columnIndex))
                        centerTY.append(dates[entryIndex]) 
                    # if not a centertooth
                    else:
                        arcTooth.append(float(arcEntry))
                        binTooth.append(float(columnIndex))
                        toothY.append(dates[entryIndex])
                # if a gap
                elif int(binEntry) == 0:
                    # if column is centerindex
                    if int(columnIndex) == 0:
                        arcCenterG.append(float(arcEntry))
                        binCenterG.append(float(columnIndex))
                        centerGY.append(dates[entryIndex])
                    # if not a centergap
                    else:
                        arcGap.append(float(arcEntry))
                        binGap.append(float(columnIndex))
                        gapY.append(dates[entryIndex])

    axFig, arcAx  = plt.subplots()
    binFig, binAx = plt.subplots()

    axFig.set_figwidth(CONFIG.WIDTH_SIZE)
    axFig.set_figheight(CONFIG.HEIGHT_SIZE)
    binFig.set_figwidth(CONFIG.WIDTH_SIZE)
    binFig.set_figheight(CONFIG.HEIGHT_SIZE)

    axFig.suptitle("Arclength Plot")
    binFig.suptitle("Index Plot")

    arcAx.scatter(arcTooth, toothY, c="c", s=10)
    arcAx.scatter(arcGap, gapY, c="#5A5A5A", s=10)
    arcAx.scatter(arcCenterG, centerGY, c="#FFCCCB", s=10)
    arcAx.scatter(arcCenterT, centerTY, c="r", s=10)

    binAx.scatter(binTooth, toothY, c="c", s=10)
    binAx.scatter(binGap, gapY, c="#5A5A5A", s=10)
    binAx.scatter(binCenterG, centerGY, c="#FFCCCB", s=10)
    binAx.scatter(binCenterT, centerTY, c="r", s=10)


    curr_date = dates[0]
    last_date = dates[-1]
    date_ticks = []
    while curr_date <= last_date:
        date_ticks.append(curr_date)
        curr_date += timedelta(days=3)

    teeth_arclength_ticks = np.linspace(0, 5000, num=101)

    arcAx.set_xticks(teeth_arclength_ticks, minor=True)
    arcAx.set_yticks(date_ticks, minor=True)
    binAx.set_yticks(date_ticks, minor=True)

    arcAx.grid(which='minor', linestyle=":")
    binAx.grid(which='minor', linestyle=":")
    arcAx.grid(which='major')
    binAx.grid(which='major')
    axFig.tight_layout()
    binFig.tight_layout()

    axFig.savefig(os.path.join(outputPath,"arclength plot.png"))
    binFig.savefig(os.path.join(outputPath,"index plot.png"))


def folder_search():
    '''
    return all csvs within the "results" folder specified in CONFIG in sorted order by path
    '''
    root = CONFIG.PATH
    allDir = [x[0] for x in os.walk(root)]
    csv = []
    
    for dir in allDir:
        items = os.listdir(dir)
        for item in items:
            if item == CONFIG.DATA_FILENAME:
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