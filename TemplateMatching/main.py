# library imports 
import os

# modules
from ImageProcessor import ImageProcessor
from utils import Match, CONFIG, suffix, print_divider

def workflow_one(images):
    # remember to change FILTER to Filter.MANUAL in CONFIG
    # for image in images:
    #     process_img = ImageProcessor(image)
    #     process_img.match(True, Match.TWO_D)
    for image in images:
        process_img = ImageProcessor(image)
        process_img.manual(True, Match.TWO_D)
    for image in images:
        process_img = ImageProcessor(image)
        process_img.filter(True)
        process_img.fit_project(True)
    ImageProcessor.plot_manual(True)

if __name__ == "__main__":
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in CONFIG.FILE_TYPES]
    workflow_one(images[0:2])
    