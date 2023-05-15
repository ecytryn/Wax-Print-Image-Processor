from enum import Enum, unique
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import cv2
from typing import Tuple


# enum classes
@unique
class Match(Enum):
    TWO_D = 1
    ONE_D = 2

@unique
class Tooth(Enum):
    TOOTH = 1
    GAP = 2
    CENTER_T = 3
    CENTER_G = 4
    NO_BOX = 5

@unique
class Filter(Enum):
    GRADIENT = 1
    GRADIENT_EVEN = 2
    SMOOTH = 3
    SMOOTH_EVEN = 4
    NONE = 5
    MANUAL = 6

@unique
class Cross(Enum):
    SQAURED = 1
    ABS = 2


# settings/configurations of the program
@dataclass(frozen=True)
class CONFIG:
    "TEMPLATE MATCHING"
    #minimum score to be considered a tooth
    THRESHOLD: float = 0.75
    THRESHOLD_1D: float = 0.75
    #permited overlap to identify two "teeth" as distinct
    IOU_THRESHOLD: float = 0.05
    IOU_THRESHOLD_1D: float = 0.05   
    #methods to use for template matching; https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html for a list of methods
    METHODS = [cv2.TM_CCOEFF_NORMED] 

    "NOISE FILTERING"
    #threshold for gradient filtering
    GRAD_THRESHOLD: float = 5
    #threshold for gradient filtering (assuming teeth are equally spaced)
    GRAD_EVEN_THRESHOLD: float = 50
    #threshold for smoothness filtering
    SMOOTH_THRESHOLD: float = 0.5
    #threshold for smoothness filtering (assuming teeth are equally spaced)
    SMOOTH_EVEN_THRESHOLD: float = 5

    "HYPERBOLA SOLVE"
    #for intensity analysis; each data point is the average over a window WINDOW_WIDTH wide
    WINDOW_WIDTH: int = 10
    #which filtering technique to choose. See class Filter for options. 
    FILTER: Filter = Filter.MANUAL
    TRANSPOSE_MANUAL = True
    
    "CROSS PROD"
    CROSS_METHOD: Cross = Cross.SQAURED
    CROSS: int = 3

    "PROJECT 1D"
    SAMPLING_WIDTH: int = 100

    "GUI"
    SQUARE = 30

    "PLOT_MANUAL"
    TIME = 3
    PATH = os.path.join(os.getcwd(),"processed", "manual data 1D")

    "OTHERS"
    # accepted filetypes to run analysis
    FILE_TYPES = [".jpg", ".png", ".jpeg"]
    #colors for manual editing; in format (G,B,R) not (R,G,B)!
    CENTER: Tuple[int, int, int] = (255,255,0) #cyan
    GAP: Tuple[int, int, int] = (0,255,255) #yellow
    TOOTH: Tuple[int, int, int] = (0,0,255) #red

    # plot style used by matplotlib
    PLOT_STYLE: str = "bmh"

    #matplotlib figure dimensions (used when output is too crammed)
    WIDTH_SIZE: int = 15
    HEIGHT_SIZE: int = 7

    # directories that will be created in /processed
    DIRS_TO_MAKE = ['match visualization', 'match data', 'match visualization 1D', 'match data 1D',
                'filter visualization', 'filter data', 'fit visualization',
                'projection', 'projection sampling', 'projection graphed', 'projection data',
                'manual data', 'manual visualization', 'manual data 1D', 'manual visualization 1D']


# helper functions
def make_dir(dir: str):
    '''create a specified directory if it doesn't already exist'''
    if not os.path.isdir(dir):
        os.mkdir(dir)

def suffix(file: str):
    '''returns the suffix of a file'''
    return os.path.splitext(file)[1]

def end_procedure():
    '''closes all current matplotlib and cv2 windows'''
    plt.close("all")
    cv2.destroyAllWindows()

def print_divider():
    '''prints a divider into the console'''
    print("============================================================")
    