#!/usr/bin/env python3
import csv
import sys
import math
import numpy as np
import cv2
import imutils
import os
import tkinter as tk
import config as cf
import rospy
import tensorflow as tf
import mrcnn.model as modellib

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as rosImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from point_cloud.msg import graspingCoor, pixelCoor
from std_msgs.msg import Bool
from PIL import Image
#from PIL import ImageTk
from skspatial.objects import Line
from skspatial.objects import Vector
from mrcnn.config import Config

from keras.backend import clear_session

try:
    #ROOT_DIR = os.path.abspath("")
    ROOT_DIR = "/home/jarvis/P6_project/src/neural_network"
    sys.path.append(ROOT_DIR)  # To find local version of the library

    ####window = tk.Tk()
    ####window.title('Detector GUI')
    ####window.geometry("1280x540+50+20")
    ####window.config(bg='#5c31ad')

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    WEIGHTS_PATH = os.path.join(ROOT_DIR, cf.modelPath)

    class CustomConfig(Config):
        NAME = "object"
        IMAGES_PER_GPU = 2
        NUM_CLASSES = 1 + 1
        STEPS_PER_EPOCH = 300
        DETECTION_MIN_CONFIDENCE = 0.98
        BATCH_SIZE = 1

    config = CustomConfig()

    CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.3

    config = InferenceConfig()

    DEVICE = "/cpu:0"
    TEST_MODE = "inference"

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    weights_path = WEIGHTS_PATH

    model.load_weights(weights_path, by_name=True)
    model.keras_model._make_predict_function()

except:
    rospy.logerr("Could not initialize Mask R-CNN")

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def detector_Callback(image, pointcloud):
    rospy.logerr("inside callback")
    pub = rospy.Publisher('/maskRCNN/graspingPoint', graspingCoor, queue_size=1)
    pub_pointcloud = rospy.Publisher('/maskRCNN/pointCloud', PointCloud2, queue_size=1)
    #try:
    cv2_img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)

    image1 = np.uint8(np.clip((cf.detectImageContrast * cv2_img + cf.detectImageBrightness),0,255))
    image1 = rotate_image(image1, 1)
    orig = image1
    image1 = image1[cf.detectImageROIyMin:cf.detectImageROIyMax, cf.detectImageROIxMin:cf.detectImageROIxMax]
    blackBorder = np.zeros((1080, 1920, 3), dtype=np.uint8)

    ROI = orig[cf.detectImageROIyMin:cf.detectImageROIyMax, cf.detectImageROIxMin:cf.detectImageROIxMax]

    x_offset = cf.detectImageROIxMin
    y_offset = cf.detectImageROIyMin
    blackBorder[y_offset:y_offset + image1.shape[0], x_offset:x_offset + image1.shape[1]] = image1

    image1 = blackBorder

        # Run object detection
    rospy.logwarn(" before model detect")
    results1 = model.detect([image1], verbose=1)
    rospy.logwarn("after model detect")

    r1 = results1[0]

    scores = r1['scores']
    mask = r1['masks']

    rospy.logerr('Results without filter: ' + str(len(scores)))

    filterVal = cf.filterValue
    maskInt = cf.maxMasks

    filtered = filter(lambda num: num > float(filterVal), scores)

    backtorgb = image1

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

            edged = cv2.Canny(opening, 50, 150)  # For edge detection / Line orientation

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

            try:
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
                            if deltaX == 0:
                                deltaX = 0.000001
                            angleInDegrees = np.arctan(deltaY / deltaX)
                            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            angles.append([angleInDegrees, line[0]])
                            lengths.append([length, line[0]])

                            # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    lengths = sorted(lengths, reverse=True)
                    rospy.logerr(str(lengths[0]))

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
            except:
                rospy.logerr('Failed to compute lines')

            try:
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
                rX, rY = int(rX), int(rY)
            except:
                rospy.logerr('Failed to compute points')

            try:
                if cf.drawCenterPoint:
                    cv2.circle(backtorgb, (cX, cY), 9, (0, 0, 255), -1)
                if cf.drawReferencePoint:
                    cv2.circle(backtorgb, (int(rX), int(rY)), 9, (0, 255, 0), -1)
                    cv2.line(backtorgb, (cX, cY), (int(rX), int(rY)), (0, 0, 0), 2)

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

                # cv2.imshow('Masked', resized)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                resized = Image.fromarray(resized)
                ####resized = ImageTk.PhotoImage(resized)
                ####panel = tk.Label(image=resized)
                ####panel.place(anchor='nw', x=315, y=-2)
                ####panel.image = resized
            except:
                rospy.logerr('Failed to show image')

    #except:
    #    rospy.logerr('Failed to compute image')


        try:
            msg = graspingCoor()
            msg.header.stamp = pointcloud.header.stamp
            msg.header.frame_id = "graspingCoordinate"
            msg.header.seq = pointcloud.header.seq
            msg.GraspingCoor = pixelCoor(x=cX, y=cY)  # grasping pixel fram MachineLearning
            msg.ReferenceCoor = pixelCoor(x=rX, y=rY)  # reference pixel/point for transformation
            pub.publish(msg)  # grasping coordinate
            pub_pointcloud.publish(pointcloud)  # pointcloud matching the image detected on

            rospy.logerr("get over it!!!")
        except:
            rospy.logerr("Cannot publish grasping coordinate")


def trigger_callback(msg):
    #try:
    tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image_color", rosImage), Subscriber("/kinect2/sd/points", PointCloud2)],queue_size=1, slop=2, allow_headerless=True)
    rospy.logerr("before callback")
    tss.registerCallback(detector_Callback)
    rospy.logerr("after callback")
    rospy.logerr(tss)
    #except:
     #   rospy.logerr("failed to syncronize '/kinect2/sd/image' and '/kinect2/sd/points'")


def main():
    rospy.init_node('MachineLearning_node', anonymous=True)
    rospy.Subscriber("/trigger", Bool, trigger_callback)
    rospy.spin()
    ####window.mainloop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("Could not start MachineLearning_node")
