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
from utils import Match, CONFIG, Filter, make_dir, suffix, end_procedure


class ImageProcessor:
    '''
    This class helps organize all the methods and data surrounding an image. 
    '''
    def __init__(self, img_name: str):
        '''
        Initialization
        '''
        self.file_type = suffix(img_name)
        self.file_name = img_name
        self.img_name = img_name.replace(self.file_type, "")

        assert os.path.isfile(os.path.join(os.getcwd(),'img', self.file_name)), f"'{self.file_name}' does not exist"
        
        self.image = cv2.imread(os.path.join('img', img_name), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]


    def match(self, display_time: bool = False, mode = Match.TWO_D):
        '''
        performs template matching
        if Match.ONE_D, matches "template 1D" images to projected image
        if Match.TWO_D, matches "template" images to original image
        '''
        start_time = time.time()
        if mode == Match.TWO_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if suffix(file) in CONFIG.FILE_TYPES]
        elif mode == Match.ONE_D:
            templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template 1D")) if suffix(file) in CONFIG.FILE_TYPES]
        try:
            template_matching.template_matching(self.file_name, self.img_name, mode, templates)
        except RuntimeError as err:
            print(err)
        if display_time and mode == Match.TWO_D:
            print(f"MATCH       | '{self.file_name}: {time.time()-start_time} s")
        if display_time and mode == Match.ONE_D:
            print(f"MATCH 1D    | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()  


    def filter(self, display_time: bool = False):
        '''

        Filter current image's data according to CONFIG.FILTER and thresholds. If it's Filter.Manual or Filter.None
        don't filter. Output result in "filter data" 
        '''
        start_time = time.time()
        if CONFIG.FILTER == Filter.MANUAL:
            path = os.path.join('processed', "manual data",f"{self.img_name}.csv")
            assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run manual first?"
        else:
            path = os.path.join('processed', "match data",f"{self.img_name}.csv")
            assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run match first?"
        noise_filtering.continuity_filter(self.file_name, self.img_name)
        if display_time:
            print(f"FILTER      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def fit_project(self, display_time: bool = False):
        '''
        takes data from "filter data" and project. If CONFIG.FILTER == Filter.MANUAL, also project manual data. 
        '''
        start_time = time.time()
        path = os.path.join('processed', "filter data", f"{self.img_name}.csv")
        img_path = os.path.join('img', self.file_name)
        assert os.path.isfile(path), f"'{self.img_name}.csv' does not exist - did you run filter first?"
        assert os.path.isfile(img_path), f"'{self.file_name}' does not exist - did you run filter first?"
        try:
            hyperbola_solve.solve(self.file_name, self.img_name, self.height)
        except RuntimeError as err:
            print(err)

        if display_time:
            print(f"FIT PROJECT | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    def manual(self, display_time: bool = False, mode = Match.ONE_D):
        '''
        runs the GUI for manual editing; 
        if Match.ONE_D, uses data from "manual 1D data" if exists, else uses data from "projection data"
        if Match.TWO_D, uses data from "manual data" if exists, else uses data from "match data"
        '''
        start_time = time.time()
        try:
            GUI.GUI(self.file_name, self.img_name, mode)
        except RuntimeError as err:
            print(err)
        if display_time:
            print(f"MANUAL      | '{self.file_name}': {time.time()-start_time} s")
        end_procedure()


    @staticmethod
    def plot_manual(display_time: bool = False):
        '''
        plot the result from "manual data 1D"
        '''
        start_time = time.time()
        try:
            plot_manual.plot_manual()
            plot_manual.plot_manual_even()
        except RuntimeError as err:
            print(err)
        if display_time:
            print(f"PLOT MANUAL | {time.time()-start_time} s")
        end_procedure()


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
make_dir("img")
make_dir("template")
make_dir("template 1D")
make_dir("processed")
os.chdir(os.path.join(current,"processed"))
for dir in CONFIG.DIRS_TO_MAKE:
    make_dir(dir)
os.chdir(current)

# suppresses warnings for a cleaner output (comment to unsuppress)
warnings.filterwarnings('ignore')
#set the theme for matplotlib plots
plt.style.use(CONFIG.PLOT_STYLE)