import os
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import csv
import sys
import json
import math
import skimage.draw
import numpy as np
import cv2
import imutils

import config as cf

from skspatial.objects import Line
from skspatial.objects import Vector

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

window = tk.Tk()

window.config(bg='#5c31ad')
# add widgets here

header = ['filename', 'type', 'threshold', 'epoch', 'expectation', 'success', 'failure']

with open('validation.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def callBack():
    image = variable.get()
    type = variable2.get()
    epochNum = epoch.get()
    thresholdNum = threshold.get()
    expectNum = expect.get()
    successNum = success.get()

    data = [image[48:], type, thresholdNum, epochNum, expectNum, successNum]
    print(data)

    with open('validation.csv', 'a', encoding='UTF8', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(data)

dirName = 'D:/Machine Learning/dataset/dataset_beumer/test'

listOfFiles = getListOfFiles(dirName)

OPTIONS = []

for elem in listOfFiles:
    OPTIONS.append(elem)

OPTIONS2 = ['box' , 'bag']

j = 75

from mrcnn.config import Config
from mrcnn import utils

import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_object_0010_1.h5")

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.98

config = CustomConfig()

CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


#LOAD MODEL
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load COCO weights, Or load the last model you trained
weights_path = WEIGHTS_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

allowable = tk.StringVar(window)
filterVal = cf.filterValue
allowable.set(filterVal)

maskVal = tk.StringVar(window)
maskInt = 1
maskVal.set(maskInt)
#allowable.trace("w", lambda name, index, mode, allowable=allowable: display_selected)

# This is for predicting images which are not present in dataset
#image_id = random.choice(dataset.image_ids)
image = os.path.join(ROOT_DIR, "dataset/dataset_beumer/test/092.jpeg")
image1 = cv2.imread(image, 1)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# resize image
#image1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)

def display_selected(choice):
    choice = variable.get()
    currently.config(text=str('Currently showing: ' + choice[48:]))
    print(choice)

    image1 = cv2.imread(choice)
    image1 = np.uint8(np.clip((cf.detectImageContrast * image1 + cf.detectImageBrightness),0,255))
    image1 = rotate_image(image1, 1)
    orig = image1
    image1 = image1[cf.detectImageROIyMin:cf.detectImageROIyMax, cf.detectImageROIxMin:cf.detectImageROIxMax]
    blackBorder = np.zeros((1080, 1920, 3), dtype=np.uint8)

    ROI = orig[cf.detectImageROIyMin:cf.detectImageROIyMax, cf.detectImageROIxMin:cf.detectImageROIxMax]

    x_offset = 500
    y_offset = 100
    blackBorder[y_offset:y_offset + image1.shape[0], x_offset:x_offset + image1.shape[1]] = image1

    image1 = blackBorder

        # Run object detection
    results1 = model.detect([image1], verbose=1)

    r1 = results1[0]

    scores = r1['scores']
    mask = r1['masks']

    print('Results without filter: ' + str(len(scores)))

    filterVal = allowable.get()
    maskInt = maskVal.get()

    scoresText.config(text=str('Detects: ' + str(len(scores))))
    filtered = filter(lambda num: num > float(filterVal), scores)
    scoresVal.config(text=str(scores))

    print(filterVal)

    backtorgb = image1

    extentArray = []
    angleArray = []
    solidityArray = []

    if mask.shape[2] != 0:
        for i in range(len(list(filtered))):
            if i == maskInt:
                break

            img = mask[:, :, i].astype('uint8')
            img *= 255
            blur = cv2.GaussianBlur(img, (cf.kernelSizeBlur, cf.kernelSizeBlur), 0)

            # Define the structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cf.kernelSizeMorphology, cf.kernelSizeMorphology))
            # Apply the opening operation
            opening = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

            edged2 = cv2.Canny(img, 50, 150)
            edged = cv2.Canny(opening, 100, 200)
            contours = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(contours)
            c = max(cnts, key=cv2.contourArea)

            if cf.drawContourRaw:
                cv2.drawContours(backtorgb, [c], -1, (255, 128, 0), 2)

            if cf.drawContourHull:
                hull = cv2.convexHull(c)
                cv2.drawContours(backtorgb, [hull], -1, (0, 255, 0), 2, 8)

            if cf.showMask:
                cv2.imshow("edge", img)

            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 5  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 30  # minimum number of pixels making up a line
            max_line_gap = 30  # maximum gap in pixels between connectable line segments

            lines = cv2.HoughLinesP(edged, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
            angles = []
            lengths = []
            if len(lines):
                for line in enumerate(lines):
                    for x1, y1, x2, y2 in line[1]:
                        deltaY = y2 - y1
                        deltaX = x2 - x1

                        angleInDegrees = np.arctan(deltaY / deltaX)
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                        angles.append([angleInDegrees, line[0]])
                        lengths.append([length, line[0]])

                        # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                print(angles)
                lengths = sorted(lengths, reverse=True)
                print(lengths)

                cAIs = []

                for k in range(len(lengths)):
                    angleMemory = round(angles[lengths[k][1]][0], 1)
                    for l in range(len(lengths)):
                        if l != 0:
                            angle = round(angles[lengths[l][1]][0], 1)
                            if angle == angleMemory:
                                coherentAngleInt = lengths[l][1]
                                cAI = coherentAngleInt
                                cAIs.append(cAI)
                                #cv2.line(backtorgb, (lines[lengths[k][1]][0][0], lines[lengths[k][1]][0][1]),
                                         #(lines[lengths[k][1]][0][2], lines[lengths[k][1]][0][3]), (255, 0, 0), 2)
                                #.line(backtorgb, (lines[cAI][0][0], lines[cAI][0][1]),
                                         #(lines[cAI][0][2], lines[cAI][0][3]), (255, 0, 0), 2)
                    cAIs.insert(0, k)
                    break


            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            line = Line.from_points(point_a=[lines[lengths[cAIs[0]][1]][0][0], lines[lengths[cAIs[0]][1]][0][1]], point_b=[lines[lengths[cAIs[0]][1]][0][2], lines[lengths[cAIs[0]][1]][0][3]])
            grasp_point = (cX, cY)
            projected_point = line.project_point(grasp_point)
            vector_projection = Vector.from_points(grasp_point, projected_point)
            vector_projection_unit = vector_projection.unit()
            magnitude = vector_projection.norm()
            rX, rY = vector_projection_unit * (magnitude / 2) + grasp_point


            # draw the contour and center of the shape on the image
            # cv2.drawContours(backtorgb, [c], -1, (0, 255, 0), 2)
            if cf.drawCenterPoint:
                cv2.circle(backtorgb, (cX, cY), 9, (0, 0, 255), -1)
            if cf.drawReferencePoint:
                cv2.circle(backtorgb, (int(rX), int(rY)), 9, (0, 255, 0), -1)
                cv2.line(backtorgb,(cX, cY),(int(rX),int(rY)),(0, 0, 0), 2)
            #cv2.circle(backtorgb, (cX, cY), 30, (255, 128, 0), 3)

            #cv2.putText(backtorgb, str(i), (cX + 30, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 128), 2)

            #cv2.putText(backtorgb, str(str(cX) + ' , ' + str(cY)), (cX - 50, cY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #cv2.putText(backtorgb, str(scores[i]), (cX - 50, cY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        #(0, 0, 255), 2)

    #cv2.imwrite('D:/Machine Learning/dataset/dataset_beumer/results/epoch4_2.jpg', backtorgb)

    red_img = np.full((1080, 1920, 3), (0, 0, 255), np.uint8)
    orig = cv2.add(orig, red_img)

    x_offset = 500
    y_offset = 100
    orig[y_offset:y_offset + ROI.shape[0], x_offset:x_offset + ROI.shape[1]] = ROI

    backtorgb = cv2.addWeighted(backtorgb, 1, orig, 0.3, 0)

    scale_percent = 50  # percent of original size
    width = int(backtorgb.shape[1] * scale_percent / 100)
    height = int(backtorgb.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(backtorgb, dim, interpolation=cv2.INTER_AREA)

    #cv2.imshow('Masked', resized)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(resized)
    resized = ImageTk.PhotoImage(resized)
    panel.configure(image=resized)
    panel.image = resized




variable = tk.StringVar(window)
variable.set(OPTIONS[0])

empty = np.zeros((1080, 1920, 3), dtype=np.uint8)
empty = Image.fromarray(empty)
empty = ImageTk.PhotoImage(empty)
panel = tk.Label(image=empty)
panel.place(anchor='nw', x=315, y=-2)

w = tk.OptionMenu(window, variable, *OPTIONS, command=display_selected)
w.place(anchor='c', x=160, y=127.5-35, width=220, height=25)

variable2 = tk.StringVar(window)

w2 = tk.OptionMenu(window, variable2, *OPTIONS2)
w2.place(anchor='c', x=160, y=127.5, width=220, height=25)

epoch = tk.StringVar(window)
a1 = tk.Entry(window, textvariable=epoch, justify='center', font=("Samsung Sharp Sans", 14, "bold"))
a1.place(anchor='c', x=220, y=90+j, width=100, height=22)

threshold = tk.StringVar(window)
a4 = tk.Entry(window, textvariable=threshold, justify='center', font=("Samsung Sharp Sans", 14, "bold"))
a4.place(anchor='c', x=220, y=112.5+j, width=100, height=22)

a5 = tk.Entry(window, textvariable=allowable, justify='center', font=("Samsung Sharp Sans", 14, "bold"), validate="focusout", validatecommand=display_selected)
a5.place(anchor='c', x=220, y=395+j, width=100, height=22)
a5.bind('<FocusOut>', display_selected)

a6 = tk.Entry(window, textvariable=maskVal, justify='center', font=("Samsung Sharp Sans", 14, "bold"), validate="focusout", validatecommand=display_selected)
a6.place(anchor='c', x=220, y=415+j, width=100, height=22)
a6.bind('<FocusOut>', display_selected)

expect = tk.StringVar(window)
a2 = tk.Entry(window, textvariable=expect, justify='center', font=("Samsung Sharp Sans", 14, "bold"))
a2.place(anchor='c', x=220, y=160+j, width=100, height=50)

success = tk.StringVar(window)
a3 = tk.Entry(window, textvariable=success, justify='center', font=("Samsung Sharp Sans", 14, "bold"))
a3.place(anchor='c', x=220, y=220+j, width=100, height=50)

button = tk.Button(window, text="Write data", command=callBack, bg='#44d460', fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
button.place(anchor='c', x=160, y=280+j, width=220, height=50)

label = tk.Label(window, text='Epochs', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 9, "bold"), bd=0)
label.place(anchor='c', x=100, y=90+j, width=100, height=25)

label = tk.Label(window, text='Threshold', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 9, "bold"), bd=0)
label.place(anchor='c', x=100, y=112.5+j, width=100, height=25)

label = tk.Label(window, text='Detector Validation', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
label.place(anchor='c', x=160, y=50, width=220, height=25)

label = tk.Label(window, text='Expected', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
label.place(anchor='c', x=100, y=160+j, width=100, height=50)

label = tk.Label(window, text='Success', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
label.place(anchor='c', x=100, y=220+j, width=100, height=50)

currently = tk.Label(window, text='Currently showing:', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 8, "bold"), bd=0, justify='left')
currently.place(anchor='c', x=160, y=320+j, width=220, height=15)

scoresText = tk.Label(window, text='Detects: ', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 8, "bold"), bd=0, justify='left')
scoresText.place(anchor='c', x=160, y=340+j, width=220, height=15)

scoresVal = tk.Label(window, text='Scores: ', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 7), bd=0, justify='left')
scoresVal.place(anchor='c', x=160, y=365+j, width=300, height=25)

label = tk.Label(window, text='Update filter', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 9, "bold"), bd=0)
label.place(anchor='c', x=100, y=395+j, width=100, height=25)

label = tk.Label(window, text='Update masks', bg='#5c31ad' ,fg='white', font=("Samsung Sharp Sans", 9, "bold"), bd=0)
label.place(anchor='c', x=100, y=415+j, width=100, height=25)

window.title('Detector GUI')
window.geometry("1280x540+50+20")
window.mainloop()
