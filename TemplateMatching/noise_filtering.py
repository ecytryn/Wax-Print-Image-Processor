import os 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def continuity_filter(data):
    THRESHOLD = 5
    WIDTH_SIZE = 10
    HEIGHT_SIZE = 5
    for csv in data:
        df = pd.read_csv(csv)
        df['x'] = df['x']+df['w']/2
        df['y'] = df['y']+df['h']/2

        x = df['x']
        y = df['y']
        gradient = np.gradient(y,x)
        df['gradient'] = gradient

        plt.style.use('bmh')
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2)
        fig.set_figwidth(WIDTH_SIZE)
        fig.set_figheight(HEIGHT_SIZE)
        fig.suptitle('Teeth and Gradient')
        ax1.plot(x,y,'.-b')
        ax2.plot(x,gradient,'.-y')

        df.drop(df[df['gradient'] > THRESHOLD].index, inplace = True)
        df.drop(df[df['gradient'] < -THRESHOLD].index, inplace = True)
        df = df[df['gradient'].notna()]
        x = df['x']
        y = df['y']
        gradient = df['gradient']
        print(gradient)
        ax3.plot(x,y,'.-b')
        ax4.plot(x,gradient,'.-y')
        plt.show()


def graph_filter():
    pass


if __name__ == "__main__":
    data = [file for file in os.listdir(os.getcwd()) if file[len(file)-4:] == ".csv"]
    continuity_filter(data)