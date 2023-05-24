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
import plot_result
from utils import Match, CONFIG, Filter, make_dir, suffix, end_procedure


class ImageProcessor:
    '''
    This class helps organize all the methods and data surrounding an image. 
    '''
    def __init__(self, file_name: str):
        '''
        Initialization
        '''
        self.root = os.getcwd()

        self.file_type = os.path.splitext(file_name)[1]
        self.file_name = file_name
        self.img_name = file_name.replace(self.file_type, "")

        assert os.path.isfile(os.path.join(os.getcwd(),'img', self.file_name)), f"'{self.file_name}' does not exist in img"
        self.image = cv2.imread(os.path.join('img', file_name), cv2.IMREAD_GRAYSCALE)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]


    def match(self, displayTime: bool = False, mode = Match.TWO_D):
        '''
        performs template matching
        if Match.ONE_D, matches "template 1D" images to projected image
        if Match.TWO_D, matches "template" images to original image
        '''
        start_time = time.time()

        target_path = os.path.join(self.root, "processed", "template matching")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if suffix(file) in CONFIG.FILE_TYPES]
        elif mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if suffix(file) in CONFIG.FILE_TYPES]
        try:
            template_matching.templateMatching(self.file_name, self.img_name, self.file_type, mode, templates)
        except RuntimeError as err:
            print(err)
        if displayTime and mode == Match.TWO_D:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        if displayTime and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()  


    def filter(self, displayTime: bool = False):
        '''
        Filter current image's data according to CONFIG.FILTER and thresholds. If it's Filter.Manual or Filter.None
        don't filter. Output result in "filter data" 
        '''
        start_time = time.time()

        target_path = os.path.join(self.root, "processed", "filter")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        if CONFIG.FILTER == Filter.MANUAL:
            path = os.path.join('processed', "manual", self.img_name,f"manual data.csv")
            assert os.path.isfile(path), f"'manual data.csv' does not exist in /processed/manual/{self.img_name} - did you run manual first?"
        else:
            path = os.path.join('processed', "template matching", self.img_name,f"template matching.csv")
            assert os.path.isfile(path), f"'template matching.csv' does not exist in /processed/template matching/{self.img_name} - did you run match first?"
        noise_filtering.continuityFilter(self.img_name, self.file_type)
        if displayTime:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def fitProject(self, displayTime: bool = False):
        '''
        takes data from "filter data" and project. If CONFIG.FILTER == Filter.MANUAL, also project manual data. 
        '''

        target_path = os.path.join(self.root, "processed", "fit")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        target_path = os.path.join(self.root, "processed", "projection")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        target_path = os.path.join(self.root, "processed", "manual")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        start_time = time.time()
        path = os.path.join('processed', "filter", self.img_name, "raw.csv")
        assert os.path.isfile(path), f"filtered files do not exist in /processed/filter/{self.img_name} - did you run filter first?"
        try:
            hyperbola_solve.solve(self.file_name, self.img_name, self.file_type, self.height)
        except RuntimeError as err:
            print(err)

        if displayTime:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def manual(self, displayTime: bool = False, mode = Match.ONE_D):
        '''
        runs the GUI for manual editing; 
        if Match.ONE_D, uses data from "manual 1D data" if exists, else uses data from "projection data"
        if Match.TWO_D, uses data from "manual data" if exists, else uses data from "match data"
        '''
        start_time = time.time()

        target_path = os.path.join(self.root, "processed", "manual")
        os.chdir(target_path)
        make_dir(self.img_name)
        os.chdir(self.root)

        try:
            GUI.GUI(self.file_name, self.img_name, self.file_type, mode)
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def plotResult(displayTime: bool = False):
        '''
        plot the result from "manual data 1D"
        '''
        start_time = time.time()
        try:
            plot_result.dataToCSV()
        except RuntimeError as err:
            print(err)
        if displayTime:
            print(f"PLOT MANUAL | {time.time()-start_time} s")
        end_procedure()


# creates the folder structure
'''
--img (images to be processed)
--template (templates for matching on images in original form)
--template 1D (templates for matching on images in strip form)
--processed (folder where results are stored)
     -- match 
     -- filter 
     -- fit 
     -- projection 
     -- manual 
'''
current = os.getcwd()
make_dir("img")
make_dir("template")
make_dir("template 1D")
make_dir("processed")
os.chdir(os.path.join(current,"processed"))
make_dir("filter")
make_dir("fit")
make_dir("template matching")
make_dir("projection")
make_dir("manual")
make_dir("output")
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(CONFIG.PLOT_STYLE)