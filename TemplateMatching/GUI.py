import cv2
import os
import pandas as pd
import numpy as np
from pynput.keyboard import Key, Controller
from utils import Tooth, Match, CONFIG


class GUI:

    MODES = [mode for mode in Tooth if mode != Tooth.NO_BOX]
    def __init__(self, fileName, imgName, fileType, mode):

        self.x = np.array([])
        self.y = np.array([])
        self.w = np.array([])
        self.h = np.array([])
        self.type = np.array([])
        self._stored_mode = Tooth.TOOTH
        self._curr_mode = Tooth.TOOTH

        # find data path
        if mode == Match.ONE_D:
            imgPath = os.path.join("processed", "projection", imgName, f"projection{fileType}")
            manualDataPath = os.path.join("processed", "manual", imgName, "manual data 1D.csv")
            if os.path.isfile(manualDataPath):
                imgDataPath = manualDataPath
            else:
                imgDataPath = os.path.join("processed", "template matching", imgName, "template matching 1D.csv")
        else:
            imgPath = os.path.join("img", fileName)
            manualDataPath = os.path.join("processed", "manual", imgName, f"manual data.csv")
            if os.path.isfile(manualDataPath):
                imgDataPath = manualDataPath
            else:
                imgDataPath = os.path.join("processed", "template matching", imgName, f"template matching.csv") 


        # if image doesn't exist, raise error and stop
        if not os.path.isfile(imgPath):
            raise RuntimeError(f"{imgPath} does not exist. Image deleted or a hyperbola fit was likely not found: did you run fitProject first?")
        
        # if data doesn't exist, use empty set, otherwise use existing data
        if not os.path.isfile(imgDataPath):
            print(f"{imgDataPath} not found, empty data set used")
        else:
            # load data
            df = pd.read_csv(imgDataPath)
            self.x = df["x"].to_numpy()
            self.y = df["y"].to_numpy()
            self.w = df["w"].to_numpy()
            self.h = df["h"].to_numpy()
            if "type" not in df.columns:
                df["type"] = [Tooth.TOOTH for _ in range(len(df.index))]
                self.type = df["type"]
            else: 
                for i in df["type"]:
                    if i == "Tooth.TOOTH":
                        self.type = np.append(self.type, Tooth.TOOTH)
                    elif i == "Tooth.GAP":
                        self.type = np.append(self.type, Tooth.GAP)
                    elif i == "Tooth.CENTER_T":
                        self.type = np.append(self.type, Tooth.CENTER_T)
                    elif i == "Tooth.CENTER_G":
                        self.type = np.append(self.type, Tooth.CENTER_G)

        # reads image, set up mouse callback
        image = cv2.imread(imgPath)
        cv2.namedWindow(imgName)
        cv2.setMouseCallback(imgName, self.leftClick)
        modeIndex = 0

        while True:

            # makes a copy of the original image 
            self.clone = image.copy()
            dataset_size = len(self.x)

            # draw data
            for i in range(dataset_size):
                self.clone = GUI.drawTooth(self.clone, self.x[i], self.y[i], self.w[i], self.h[i], self.type[i])

            # displays current mode
            if self._curr_mode == Tooth.TOOTH:
                textImg = cv2.putText(self.clone, "mode: tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.TOOTH, 2)
            elif self._curr_mode == Tooth.GAP: 
                textImg = cv2.putText(self.clone, "mode: gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.GAP, 2)
            elif self._curr_mode == Tooth.CENTER_T: 
                textImg = cv2.putText(self.clone, "mode: center tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
            elif self._curr_mode == Tooth.CENTER_G:
                textImg = cv2.putText(self.clone, "mode: center gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, CONFIG.CENTER, 2)
            else:
                textImg = cv2.putText(self.clone, "mode: viewing", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100,100,100), 2)

            cv2.imshow(imgName, textImg)

            # logic for key presses
            key = cv2.waitKey(0)

            # space
            if key == 32:
                # if in viewing mode already, store mode 
                if self._curr_mode == Tooth.NO_BOX:
                    self._curr_mode = self._stored_mode
                else:
                    # else store current mode, change into viewing mode
                    self._stored_mode = self._curr_mode
                    self._curr_mode = Tooth.NO_BOX
            # esc
            elif key == 27:
                cv2.destroyAllWindows()
                break
            # tab
            elif key == 9 and self.MODE != Tooth.NO_BOX:
                # change into the next mode
                modeIndex = (modeIndex + 1) % len(self.MODES)
                self._curr_mode = self.MODES[modeIndex]
            # s
            elif key == ord("s"):
                dfRes = pd.DataFrame()
                dfRes["x"] = self.x
                dfRes["y"] = self.y
                dfRes["w"] = self.w
                dfRes["h"] = self.h
                dfRes["type"] = self.type
                dfRes.sort_values(by=["x"], inplace=True)
                GUI.save(fileName, imgName, fileType, mode, self.clone, dfRes)
                break
            # 1
            elif key == ord("1"):
                self.MODE = Tooth.TOOTH
                modeIndex = 0
            # 2
            elif key == ord("2"):
                self.MODE = Tooth.GAP
                modeIndex = 1
            # 3
            elif key == ord("3"):
                self.MODE = Tooth.CENTER_T
                modeIndex = 2
            # 4
            elif key == ord("4"):
                self.MODE = Tooth.CENTER_G
                modeIndex = 3

    @staticmethod
    def drawTooth(image, x, y, w, h, type):
        OFFSET_X = -5
        OFFSET_Y = 5

        centerX = int(x + 1/2 * w)
        centerY = int(y + 1/2 * h)
        endX = x + w
        endY = y + h

        if type == Tooth.TOOTH:
            color = CONFIG.TOOTH
        elif type == Tooth.GAP:
            color = CONFIG.GAP
        elif type == Tooth.CENTER_T:
            color = CONFIG.CENTER
            imageLabelled = GUI._drawLabel(image, centerX+OFFSET_X, centerY+OFFSET_Y, color, "T")
        elif type == Tooth.CENTER_G:
            color = CONFIG.CENTER
            imageLabelled = GUI._drawLabel(image, centerX+OFFSET_X, centerY+OFFSET_Y, color, "G")
        imageLabelled = GUI._drawCenter(image, centerX, centerY, color)

        if type == Tooth.NO_BOX:
            return imageLabelled

        imageRec = GUI._drawRectangle(imageLabelled, x, y, endX, endY, color)
        return imageRec
    
    @staticmethod
    def _drawRectangle(image, x, y, endX, endY, COLOR):
        THICKNESS = 2
        newImage = cv2.rectangle(image, pt1=(x,y), pt2=(endX,endY), color=COLOR, thickness=2)
        return newImage
    
    @staticmethod
    def _drawCenter(image, centerX, centerY, COLOR):
        newImage = cv2.circle(image, (centerX, centerY), radius=5, color=COLOR, thickness=-1)
        return newImage
    
    @staticmethod
    def _drawLabel(image, x, y, color, label):
        newImage = cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        return newImage


    def leftClick(self, event, clickedX, clickedY, flags, params):
        if event == cv2.EVENT_LBUTTONUP and self._curr_mode != Tooth.NO_BOX:
            draw = True
            dataset_size = len(self.x)

            for index in range(dataset_size):
                # if the user clicks within a square, delete the square 
                if (clickedX >= self.x[index] and clickedX <= self.x[index]+self.w[index] 
                    and clickedY >= self.y[index] and clickedY <= self.y[index]+self.h[index]):
                    self._delete_index_data(index)
                    draw = False
                    break
            
            # if the user clicks a new center, delete the old one
            if draw and (self._curr_mode == Tooth.CENTER_T or self._curr_mode == Tooth.CENTER_G):
                centerT = np.where(type == Tooth.CENTER_T)
                for index in centerT[0]:
                    self._delete_index_data(index)
                centerG = np.where(type == Tooth.CENTER_G)
                for index in centerG[0]:
                    self._delete_index_data(index)

            # otherwise, draw a square at the clicked location
            if draw:
                newX = int(clickedX - 1/2 * CONFIG.SQUARE)
                newY = int(clickedY - 1/2 * CONFIG.SQUARE)
                self.x = np.append(self.x, newX)
                self.y = np.append(self.y, newY)
                self.w = np.append(self.w, CONFIG.SQUARE)
                self.h = np.append(self.h, CONFIG.SQUARE)
                self.type = np.append(self.type, self._curr_mode)

            # this section tricks the GUI into reloading by pressing "a"
            keyboard = Controller()
            keyboard.press("a")
            keyboard.release("a")

    @staticmethod
    def save(fileName, imgName, fileType, mode, image, dfRes):
        if mode == Match.ONE_D:  
            PATH_IMG = os.path.join("processed", "manual", imgName, f"manual 1D{fileType}")
            PATH_DATA = os.path.join("processed", "manual", imgName, f"manual data 1D.csv")
        else:
            PATH_IMG = os.path.join("processed", "manual", imgName, f"manual{fileType}")
            PATH_DATA = os.path.join("processed", "manual", imgName, f"manual data.csv")

        cv2.imwrite(PATH_IMG, image)
        dfRes.to_csv(PATH_DATA)
        cv2.destroyAllWindows()

    def _delete_index_data(self, index):
        self.x = np.delete(self.x, index)
        self.y = np.delete(self.y, index)
        self.w = np.delete(self.w, index)
        self.h = np.delete(self.h, index)
        self.type = np.delete(self.type, index)


def plotPreviousData(fileName, imgName, fileType, mode, df):
    if mode == Match.ONE_D:
        imgPath = os.path.join("processed", "projection", imgName, f"projection{fileType}")
    else:
        imgPath = os.path.join("img", fileName)

    # reads dataframe
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    w = df["w"].to_numpy()
    h = df["h"].to_numpy()
    type = np.array([])
    for i in df["type"]:
        if i == "Tooth.TOOTH":
            type = np.append(type, Tooth.TOOTH)
        elif i == "Tooth.GAP":
            type = np.append(type, Tooth.GAP)
        elif i == "Tooth.CENTER_T":
            type = np.append(type, Tooth.CENTER_T)
        elif i == "Tooth.CENTER_G":
            type = np.append(type, Tooth.CENTER_G)

    if not os.path.isfile(imgPath):
        raise RuntimeError(f"{imgPath} does not exist. A hyperbola fit was likely not found or did you run fitProject first?")

    image = cv2.imread(imgPath)
    for i in range(len(x)):
        image = GUI.drawTooth(image, int(x[i]-1/2*w[i]), int(y[i]-1/2*h[i]), w[i], h[i], type[i])

    GUI.save(fileName, imgName, fileType, mode, image, df)