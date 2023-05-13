# library imports 
import os

# modules
from ImageProcessor import ImageProcessor
from utils import Match, CONFIG, suffix, print_divider

if __name__ == "__main__":
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in CONFIG.FILE_TYPES]
    print_divider()
    for image in images[0:1]:
        process_img = ImageProcessor(image)
        process_img.match(True, Match.TWO_D)
        process_img.manual(True, Match.TWO_D)
        print_divider()