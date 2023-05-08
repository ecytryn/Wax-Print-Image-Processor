# library imports
import os 
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#helper functions
import template_matching
import noise_filtering
import hyperbola_solve
import project_1D

class ImageProcessor:
    
    def __init__(self, img_name: str, file_type: str = None, plot_style: str = None):
        
        if file_type: 
            self.file_type = file_type
        else: 
            self.file_type = '.jpg'
        self.name = img_name[:len(img_name)-4]

        assert os.path.isdir('img'), 'img directory does not exist'
        self.image = cv2.imread(os.path.join('img', img_name), cv2.IMREAD_GRAYSCALE)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        #set the theme for matplotlib plots
        if plot_style:
            plt.style.use(plot_style)
        else: 
            plt.style.use('bmh')

        current = os.getcwd()
        self.make_dir("processed")
        os.chdir(os.path.join(current,"processed"))
        self.make_dir("match visualization")
        self.make_dir("match data")
        self.make_dir("filter visualization")
        self.make_dir("fit visualization")
        self.make_dir("projection")
        self.make_dir("projection_sampling")
        self.make_dir("projection gradient")
        self.make_dir("projection graphed")
        os.chdir(current)

    @staticmethod
    def make_dir(name: str):
        if not os.path.isdir(name):
            os.mkdir(name)

    def match(self, displayTime: bool = False, plot: bool = False):
        start_time = time.time()
        templates = [file for file in os.listdir(os.path.join(os.getcwd(),"template")) if file[len(file)-4:] == self.file_type]
        template_matching.template_matching(f"{self.name}{self.file_type}", templates, 0.75, 0.05, plot)
        if displayTime:
            print(f"MATCH | '{self.name}{self.file_type}': {time.time()-start_time} s")
    

    def filter(self, gradthreshold: float = 5, gradeventhreshold: float = 50,
               smooththreshold: float = 0.5, smootheventhreshold: float = 5,
               displayTime: bool = False, plot: bool = False):
        start_time = time.time()
        noise_filtering.continuity_filter()
        if displayTime:
            print(f"FILTER | '{self.name}.{self.file_type}': {time.time()-start_time} s")


    def fit(self, displayTime: bool = False, plot: bool = False):
        start_time = time.time()
        if displayTime:
            print(f"FIT | '{self.name}.{self.file_type}': {time.time()-start_time} s")


    def project(self, displayTime: bool = False, plot: bool = False):
        start_time = time.time()
        if displayTime:
            print(f"PROJECT | '{self.name}.{self.file_type}': {time.time()-start_time} s")




if __name__ == "__main__":
    FILETYPE = ".jpg"
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if file[len(file)-4:] == FILETYPE]
    for image in images:
        processing = ImageProcessor(image, FILETYPE)
        processing.match(True, False)
