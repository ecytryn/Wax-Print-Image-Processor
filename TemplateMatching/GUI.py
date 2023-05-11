import cv2
import os
import pandas as pd
import numpy as np
from dataclass import Tooth
from pynput.keyboard import Key, Controller


# I think it's BGR for some reason, not RGB
GAP_YELLOW=(0,255,255)
TOOTH_RED=(0,0,255)
GREY = (220,220,220)
REC_THICKNESS = 2
WIN_NAME = "Show me those baby whites!"
SQUARE = 30
MODE = Tooth.TOOTH

x = []
y = []
w = []
h = []
type = []
clone = 0


def GUI(FILE_NAME, NAME):

    global x, y, w, h, type, clone, MODE

    IMG_PATH = os.path.join("processed", "projection", FILE_NAME)
    IMG_DATA_PATH = os.path.join("processed", "match data 1D", f"{NAME}.csv")

    if not os.path.isfile(IMG_PATH) or not os.path.isfile(IMG_DATA_PATH):
        raise RuntimeError(f"{FILE_NAME} or {NAME} does not exist. A hyperbola fit was likely not found or did you run fit_project first?")

    image = cv2.imread(IMG_PATH)
    df = pd.read_csv(IMG_DATA_PATH)
    df["type"] = [Tooth.TOOTH for _ in range(len(df.index))]

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    w = df["w"].to_numpy()
    h = df["h"].to_numpy()
    type = df["type"].to_numpy()

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, left_click)

    while 1:
        clone = image.copy()
        for i in range(len(x)):
            draw_tooth(clone, x[i], y[i], w[i], h[i], type[i])

        # border_img = cv2.copyMakeBorder(clone, 1000, 1000, 0, 0, cv2.BORDER_CONSTANT, value=GREY)
        if MODE == Tooth.TOOTH:
            text_img = cv2.putText(clone, "mode: tooth", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, TOOTH_RED, 2)
        else: 
            text_img = cv2.putText(clone, "mode: gap", (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, GAP_YELLOW, 2)
        cv2.imshow(WIN_NAME, text_img)
        key = cv2.waitKey(0)
        if key == 32:
            cv2.destroyAllWindows()
            break
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == 9:
            if MODE == Tooth.GAP:
                MODE = Tooth.TOOTH
            else:
                MODE = Tooth.GAP
        if key == ord("s"):
            save(FILE_NAME, NAME)
            break

def draw_tooth(image, x, y, w, h, MODE):
    center_x = int(x + 1/2 * w)
    center_y = int(y + 1/2 * h)
    end_x = x + w
    end_y = y + h
    if MODE == Tooth.TOOTH:
        color = TOOTH_RED
    else:
        color = GAP_YELLOW
    image_rec = draw_rectangle(image, x, y, end_x, end_y, color)
    image_cen = draw_center(image_rec, center_x, center_y, color)
    return image_cen

def draw_rectangle(image, x, y, end_x, end_y, COLOR):
    THICKNESS = 2
    new_img = cv2.rectangle(image, pt1=(x,y), pt2=(end_x,end_y), color=COLOR, thickness=REC_THICKNESS)
    return new_img

def draw_center(image, center_x, center_y, COLOR):
    new_img = cv2.circle(image, (center_x, center_y), radius=5, color=COLOR, thickness=-1)
    return new_img

def left_click(event, clicked_x, clicked_y, flags, params):

    global x, y, w, h, type
    if event == cv2.EVENT_LBUTTONUP:
        draw = True
        for index in range(len(x)):
            if clicked_x >= x[index] and clicked_x <= x[index]+w[index] and clicked_y >= y[index] and clicked_y <= y[index]+h[index]:
                x = np.delete(x, index)
                y = np.delete(y, index)
                w = np.delete(w, index)
                h = np.delete(h, index)
                type = np.delete(type, index)
                draw = False
                break
        if draw:
            new_x = int(clicked_x - 1/2 * SQUARE)
            new_y = int(clicked_y - 1/2 * SQUARE)
            x = np.append(x, new_x)
            y = np.append(y, new_y)
            w = np.append(w, SQUARE)
            h = np.append(h, SQUARE)
            type = np.append(type, MODE)
        keyboard = Controller()
        keyboard.press("a")
        keyboard.release("a")

def save(FILE_NAME, NAME):
    PATH_IMG = os.path.join("processed", "manual visualization", FILE_NAME)
    PATH_DATA = os.path.join("processed", "manual data", f"{NAME}.csv")
    cv2.imwrite(PATH_IMG, clone)
    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y
    df["w"] = w
    df["h"] = h
    df["type"] = type
    df.to_csv(PATH_DATA)
    cv2.destroyAllWindows()

