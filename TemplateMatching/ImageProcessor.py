# library imports
import os 
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings

#helper functions
import template_matching
import noise_filtering
import hyperbola_solve
import GUI
import plot_manual
from utils import Match, CONFIG, Filter, makeDir, suffix, endProcedure


class ImageProcessor:
    '''
    This class helps organize all the methods and data surrounding an image. 
    '''
    def __init__(self, imgName: str):
        '''
        Initialization
        '''
        self.fileType = suffix(imgName)
        self.fileName = imgName
        self.imgName = imgName.replace(self.fileType, "")

        assert os.path.isfile(os.path.join(os.getcwd(),'img', self.fileName)), f"'{self.fileName}' does not exist"
        
        self.image = cv2.imread(os.path.join('img', imgName), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]


    def match(self, displayTime: bool = False, mode = Match.TWO_D):
        '''
        performs template matching
        if Match.ONE_D, matches "template 1D" images to projected image
        if Match.TWO_D, matches "template" images to original image
        '''
        startTime = time.time()
        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if suffix(file) in CONFIG.FILE_TYPES]
        elif mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if suffix(file) in CONFIG.FILE_TYPES]
        try:
            template_matching.templateMatching(self.fileName, self.imgName, mode, templates)
        except RuntimeError as err:
            print(err)
        if displayTime and mode == Match.TWO_D:
            print(f"MATCH       | '{self.fileName}: {time.time()-startTime} s")
        if displayTime and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.fileName}': {time.time()-startTime} s")
        endProcedure()  


    def filter(self, displayTime: bool = False):
        '''

        Filter current image's data according to CONFIG.FILTER and thresholds. If it's Filter.Manual or Filter.None
        don't filter. Output result in "filter data" 
        '''
        startTime = time.time()
        if CONFIG.FILTER == Filter.MANUAL:
            path = os.path.join('processed', "manual data",f"{self.imgName}.csv")
            assert os.path.isfile(path), f"'{self.imgName}.csv' does not exist - did you run manual first?"
        else:
            path = os.path.join('processed', "match data",f"{self.imgName}.csv")
            assert os.path.isfile(path), f"'{self.imgName}.csv' does not exist - did you run match first?"
        noise_filtering.continuityFilter(self.fileName, self.imgName)
        if displayTime:
            print(f"FILTER      | '{self.fileName}': {time.time()-startTime} s")
        endProcedure()


    def fitProject(self, displayTime: bool = False):
        '''
        takes data from "filter data" and project. If CONFIG.FILTER == Filter.MANUAL, also project manual data. 
        '''
        startTime = time.time()
        path = os.path.join('processed', "filter data", f"{self.imgName}.csv")
        imgPath = os.path.join('img', self.fileName)
        assert os.path.isfile(path), f"'{self.imgName}.csv' does not exist - did you run filter first?"
        assert os.path.isfile(imgPath), f"'{self.fileName}' does not exist - did you run filter first?"
        try:
            hyperbola_solve.solve(self.fileName, self.imgName, self.height)
        except RuntimeError as err:
            print(err)

        if displayTime:
            print(f"FIT PROJECT | '{self.fileName}': {time.time()-startTime} s")
        endProcedure()


    def manual(self, displayTime: bool = False, mode = Match.ONE_D):
        '''
        runs the GUI for manual editing; 
        if Match.ONE_D, uses data from "manual 1D data" if exists, else uses data from "projection data"
        if Match.TWO_D, uses data from "manual data" if exists, else uses data from "match data"
        '''
        startTime = time.time()
        try:
            GUI.GUI(self.fileName, self.imgName, mode)
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"MANUAL      | '{self.fileName}': {time.time()-startTime} s")
        endProcedure()


    @staticmethod
    def plotManual(displayTime: bool = False):
        '''
        plot the result from "manual data 1D"
        '''
        startTime = time.time()
        try:
            plot_manual.plotManual()
            plot_manual.plotManualEven()
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"PLOT MANUAL | {time.time()-startTime} s")
        endProcedure()


# creates the folder structure
'''
--img (images to be processed)
--template (templates for matching on images in original form)
--template 1D (templates for matching on images in strip form)
--processed (folder where results are stored)
     -- match visualization (template matching result overlapped on original image)
     -- match data (result data of template matching)
     -- match visualization 1D (1D template matching result overlapped on projected image)
     -- match data 1D (result data of 1D template matching)
     -- filter visualization (visualization of the 4 filtering techniques)
     -- filter data (result data after filtering)
     -- fit visualization (fit conic overlapped on original image)
     -- projection (projection result visualization)
     -- projection sampling (visualization of the sampling done by projection)
     -- projection graphed (intensity analysis visualization of projection)
     -- projection data (data of projected points along conic and orthogonal vectors)
     -- manual visualization (manual editing result overlapped on original image)
     -- manual data (data of manual editing result)
     -- manual visualization 1D (1D manual editing result overlapped on 1D image)
     -- manual data 1D (data of 1D manual editing result)
'''
current = os.getcwd()
makeDir("img")
makeDir("template")
makeDir("template 1D")
makeDir("processed")
os.chdir(os.path.join(current,"processed"))
for dir in CONFIG.DIRS_TO_MAKE:
    makeDir(dir)
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(CONFIG.PLOT_STYLE)