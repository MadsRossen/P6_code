#!/usr/bin/env python3
import csv
from gc import collect
import sys
import math
from grpc import Status
import numpy as np
import cv2
import imutils
import os
import tkinter as tk
import config as cf
import rospy
import tensorflow as tf
import mrcnn.model as modellib

import threading


from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as rosImage

from message_filters import Subscriber, ApproximateTimeSynchronizer
from point_cloud.msg import graspingCoor, pixelCoor
from std_msgs.msg import Header
from PIL import Image
from PIL import ImageTk
from skspatial.objects import Line
from skspatial.objects import Vector
from mrcnn.config import Config
from keras.backend import clear_session

from point_cloud.msg import graspingCoor, planePoints
import ros_numpy
import math
import random 
from geometry_msgs.msg import Pose, Point #Point is for testing without planePoints msg
from plane_creator.msg import TransRot

from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.objects import Point as scikitPoint
from skspatial.plotting import plot_3d

from scipy.spatial.transform import Rotation
from point_cloud.msg import planePoints


try:
    rospy.init_node('MachineLearning_node', anonymous=True)

    ROOT_DIR = "/home/jarvis/P6_project/src/neural_network"
    sys.path.append(ROOT_DIR)  # To find local version of the library

    window = tk.Tk()
    window.config(bg='#003659')

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
    starttxt = str(rospy.Time.now().to_sec()) + '\n'
    with open('startTime.txt', 'a') as f:
        f.write(starttxt)
        
    #rospy.logerr("inside callback")
    
    #try: 
    cv2_img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    #cv2.imwrite("raw.jpg", cv2_img)
    global status
    status.config(text='Image is loaded')
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
    #filename = str(rospy.Time.now()) + 'image1.jpg'
    #rospy.logerr(filename)
    #cv2.imwrite(filename, image1)

        # Run object detection
    
    status.config(text='Detection is starting up')
#    rospy.logwarn(" before model detect")
    results1 = model.detect([image1], verbose=1)
#    rospy.logwarn("after model detect")

    r1 = results1[0]

    scores = r1['scores']
    mask = r1['masks']

    rospy.logerr('Results without filter: ' + str(len(scores)))
#    rospy.logwarn(scores)

    filterVal = cf.filterValue
    maskInt = cf.maxMasks

    filtered = filter(lambda num: num > float(filterVal), scores)

    backtorgb = image1
    
    status.config(text='Finding masks. Maximum ' + str(maskInt) + ' will be found')
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
#                    rospy.logerr(str(lengths[0]))

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
            
            
            status.config(text='Centerpoint has been found')

            try:
                if cf.drawCenterPoint:
                    cv2.circle(backtorgb, (cX, cY), 9, (0, 0, 255), -1)
                if cf.drawReferencePoint:
                    cv2.circle(backtorgb, (int(rX), int(rY)), 9, (0, 255, 0), -1)
                    cv2.line(backtorgb, (cX, cY), (int(rX), int(rY)), (0, 0, 0), 2)

                red_img = np.full((1080, 1920, 3), (0, 0, 255), np.uint8)
                orig = cv2.add(orig, red_img)

                x_offset = cf.detectImageROIxMin
                y_offset = cf.detectImageROIyMin
                orig[y_offset:y_offset + ROI.shape[0], x_offset:x_offset + ROI.shape[1]] = ROI

                backtorgb = cv2.addWeighted(backtorgb, 1, orig, 0.3, 0)

                scale_percent = 50  # percent of original size
                width = int(backtorgb.shape[1] * scale_percent / 100)
                height = int(backtorgb.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                resized = cv2.resize(backtorgb, dim, interpolation=cv2.INTER_AREA)
                filename1 = str(rospy.Time.now()) + 'Masked.jpg'
                cv2.imwrite(filename1, backtorgb)    

                backtorgb = backtorgb[ : , 215:1190+630]

                #cv2.imshow('Masked', resized)
                backtorgb = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2RGB)
                backtorgb = Image.fromarray(backtorgb)
                backtorgb = ImageTk.PhotoImage(backtorgb)
                #panel = tk.Label(image=empty)
                #panel.place(anchor='nw', x=315, y=-2)
                global panel
                panel.configure(image=backtorgb)
                panel.image = backtorgb
            except:
                rospy.logerr('Failed to show image')

    #except:
    #    rospy.logerr('Failed to compute image')

        #try:
        msg = graspingCoor()
        msg.header.stamp = pointcloud.header.stamp
        msg.header.frame_id = "graspingCoordinate"
        msg.header.seq = pointcloud.header.seq
        msg.GraspingCoor = pixelCoor(x=cX, y=cY)  # grasping pixel fram MachineLearning
        msg.ReferenceCoor = pixelCoor(x=rX, y=rY)  # reference pixel/point for transformation
        #msg.GraspingCoor = pixelCoor(x=860, y=480)
        #msg.ReferenceCoor = pixelCoor(x=860, y=400)
        pointcloudCallback(msg, pointcloud)
        #except:
        #    rospy.logerr("Cannot publish grasping coordinate")





def pointcloudCallback(grasp, points):    

    # x0 = grasp.GraspingCoor.x
    # y0 = grasp.GraspingCoor.y
    # xref = grasp.ReferenceCoor.x
    # yref = grasp.ReferenceCoor.y

    x0 = int(grasp.GraspingCoor.x / (1920/512))
    y0 = int(grasp.GraspingCoor.y /  (1080/424))
    xref = int(grasp.ReferenceCoor.x / (1920/512))
    yref = int(grasp.ReferenceCoor.y / (1080/424))

    liste = [x0, y0, xref, yref]
    rospy.logerr(liste)

    #defines how large an area to forward
    searchSpace = 5
    size = len(range(-searchSpace, searchSpace)) * len(range(-searchSpace, searchSpace))
    set = []  
    data = ros_numpy.numpify(points)

    for i in range(-searchSpace, searchSpace):
        for j in range (-searchSpace, searchSpace):
            if   math.isnan(data[y0+i][x0+j][0]) or math.isnan(data[y0+i][x0+j][1]) or math.isnan(data[y0+i][x0+j][2]):
                pass
            else:
                set.append(Point(x = data[y0+i][x0+j][0], y = data[y0+i][x0+j][1], z = data[y0+i][x0+j][2]))
                
    # length of points published on topic
    len_of_data = len(set) -2

#    rospy.logerr(len(set))
    # random list of unique number 
    #try:
    ran_list = random.sample(range(1, len(set)), len_of_data)
    pointData = []
    for i in range (len(ran_list)):
            pointData.append(set[ran_list[i]])
    msg = planePoints()
    msg.GraspingPoint = Point(x = data[y0][x0][0], y = data[y0][x0][1], z = data[y0][x0][2])
    msg.ReferencePoint = Point(x = data[yref][xref][0], y = data[yref][xref][1], z = data[yref][xref][2])
    msg.PlanePoints = pointData
    ConvertCallback(msg)
    #except:
    #    rospy.logerr("Shit list is too short")

##Converts the data from message frompoints to something we can work with
def ConvertCallback(msg):
    pointListX = []
    pointListY = []
    pointListZ = []

    ## Actual points
    DataPoints = msg.PlanePoints
    grasp = msg.GraspingPoint
    ref = msg.ReferencePoint

    ListOfPoints = []
    for point in DataPoints:
        ListOfPoints.append(ros_numpy.numpify(point))

    maxPointNum = 50
    if len(ListOfPoints) > maxPointNum:
        for i in range(0,maxPointNum):
            if math.isnan(ListOfPoints[i][0]) != True:
                pointListX.append(ListOfPoints[i][0])
            if math.isnan(ListOfPoints[i][1]) != True:
                pointListY.append(ListOfPoints[i][1])
            if math.isnan(ListOfPoints[i][2]) != True:
                pointListZ.append(ListOfPoints[i][2])
    elif len(ListOfPoints) <= maxPointNum:
        for point in ListOfPoints:
            if math.isnan(point[0]) != True:
                pointListX.append(point[0])
            if math.isnan(point[1]) != True:
                pointListY.append(point[1])
            if math.isnan(point[2]) != True:
                pointListZ.append(point[2])

    MakePointList(pointListX, pointListY, pointListZ, grasp, ref)

##Makes a list of sk.spatial points based on the geometry_msgs points 
def MakePointList(pointListX, pointListY, pointListZ, grasp, ref):
    tempPointList = []
    for i in range(len(pointListX)):
        tempPoint = []
        for j in range(3):
            if j == 0:
                tempPoint.append(pointListX[i])
            if j == 1: 
                tempPoint.append(pointListY[i])
            if j == 2:
                tempPoint.append(pointListZ[i])
        tempPointList.append(tempPoint)
    pointList = Points(tempPointList)
    CreatePlane(pointList, grasp, ref)

##Creates a plane based on the sk.spatial points
def CreatePlane(pointList, grasp, ref):
    plane = Plane.best_fit(pointList)
    ref_Point = scikitPoint([ref.x, ref.y, ref.z])
    ref_projected = plane.project_point(ref_Point)
    PlanePointAndNormal(plane, grasp, ref_projected)

## split the plane to the centroid point and normal vector
def PlanePointAndNormal(plane, grasp, ref_projected):
    centroid = []
    centroid.append(plane.point[0])
    centroid.append(plane.point[1])
    centroid.append(plane.point[2])

    normvector = []
    normvector.append(plane.normal[0])
    normvector.append(plane.normal[1])
    normvector.append(plane.normal[2])

    ConvertToPose(centroid, normvector, grasp, ref_projected)

## Converts centroid and normal vector to geometry_msg/Pose (point and quaternions)
def ConvertToPose(centroid, normvector, grasp, ref_projected):
    msg = TransRot()
    msg.translationVector[0] = (grasp.x * -1) - 0.085 
    msg.translationVector[1] = grasp.y + 0.52248
    msg.translationVector[2] = 1.0271 - (grasp.z + 0.008)

    #rospy.logerr(1.0271-(self.grasp.z + 0.008))

    xVectorCam = [1, 0, 0]
    yVectorCam = [0, 1, 0]
    zVectorCam = [0, 0, 1]

    xPunkt = ref_projected

    xVector = [centroid[0]+xPunkt[0], centroid[1]+xPunkt[1], centroid[2]+xPunkt[2]]
    xVector = xVector/np.linalg.norm(xVector)
    zVector = normvector
    yVector = np.cross(xVector, zVector)

    Q11 = np.arccos(np.dot(xVectorCam, xVector))
    Q12 = np.arccos(np.dot(xVectorCam, yVector))
    Q13 = np.arccos(np.dot(xVectorCam, zVector))

    Q21 = np.arccos(np.dot(yVectorCam, xVector))
    Q22 = np.arccos(np.dot(yVectorCam, yVector))
    Q23 = np.arccos(np.dot(yVectorCam, zVector))

    Q31 = np.arccos(np.dot(zVectorCam, xVector))
    Q32 = np.arccos(np.dot(zVectorCam, yVector))
    Q33 = np.arccos(np.dot(zVectorCam, zVector))

    msg.rotationMatrix[0] = Q11
    msg.rotationMatrix[1] = Q12
    msg.rotationMatrix[2] = Q13
    msg.rotationMatrix[3] = Q21
    msg.rotationMatrix[4] = Q22
    msg.rotationMatrix[5] = Q23
    msg.rotationMatrix[6] = Q31
    msg.rotationMatrix[7] = Q32
    msg.rotationMatrix[8] = Q33

    PublishPlane(msg)

## Publishes the grasping pose for the plane in relation to the camera
def PublishPlane(msg):
    pub = rospy.Publisher('/grasping_pose', TransRot, queue_size=1)
    pub.publish(msg)
#    rospy.logwarn(msg)
#    rospy.logerr("Grasping pose published")
    global status
    status.config(text='Grasping pose published')

from pynput import keyboard as ky
def on_press(key):
    if key == ky.Key.space:
        global spacePres 
        spacePres = True
    else:
        return False

def on_release(key):
    if key == ky.Key.space:
        return False

def collectImages(image, pointcloud):
    global imageReturn
    global pointReturn
    pointReturn = pointcloud
    imageReturn = image
    return imageReturn, pointReturn 

class App(threading.Thread):

    def __init__(self, tk_root):
        self.root = tk_root
        threading.Thread.__init__(self)
        self.start()
    
    def run(self):
        loop_active = True
        while loop_active:
            thread()

def buttonRun():
    global status
    status.config(text='Running')
    global imageReturn
    global pointReturn
    detector_Callback(imageReturn, pointReturn)

def buttonReset():
    print("Reset")
    global status
    status.config(text='Reset commencing...')
    # Should move the robot to drop point, no matter what

def buttonStop():
    print("E-STOP")
    global status
    status.config(text='Stopping')
    #Kill the robot

def thread():
    if not rospy.is_shutdown():
        rate = rospy.Rate(10) # 10hz
        #NeuralNetwork()
        #rospy.logerr("Could not start MachineLearning_node")
        tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image_color", rosImage), Subscriber("/kinect2/sd/points", PointCloud2)], queue_size=1,slop=0.1)
        tss.registerCallback(collectImages)
        #try:

        global imageReturn
        global pointReturn

        with ky.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

            if spacePres == True:
                #tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image_color", rosImage), Subscriber("/kinect2/sd/points", PointCloud2)], queue_size=1,slop=0.1)
                #tss.registerCallback(detector_Callback)
                detector_Callback(imageReturn, pointReturn)
            #except:
            #rospy.logerr("failed to syncronize '/kinect2/hd/image' and '/kinect2/sd/points'")
        rate.sleep()

def gui():
    empty = np.zeros((1080, 1520, 3), dtype=np.uint8)
    empty = Image.fromarray(empty)
    empty = ImageTk.PhotoImage(empty)
    global panel
    panel = tk.Label(image=empty)
    panel.place(anchor='nw', x=400, y=-2)

    label = tk.Label(window, text='Control Panel', bg='#003659' ,fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
    label.place(anchor='c', x=200, y=50, width=220, height=25)

    stat = tk.Label(window, text='Operation status:', bg='#003659' ,fg='white', font=("Samsung Sharp Sans", 8, "bold"), bd=0, justify='left')
    stat.place(anchor='c', x=200, y=85, width=220, height=15)

    global status
    status = tk.Label(window, text='', bg='#003659' ,fg='white', font=("Samsung Sharp Sans", 10, "bold"), bd=0, justify='left')
    status.place(anchor='c', x=200, y=100, width=220, height=15)

    button = tk.Button(window, text="RUN", command=buttonRun, bg='#44d460', fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
    button.place(anchor='c', x=200, y=200, width=220, height=50)

    button = tk.Button(window, text="RESET", command=buttonReset, bg='#ebb134', fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
    button.place(anchor='c', x=200, y=265, width=220, height=50)

    button = tk.Button(window, text="STOP", command=buttonStop, bg='#e62525', fg='white', font=("Samsung Sharp Sans", 14, "bold"), bd=0)
    button.place(anchor='c', x=200, y=330, width=220, height=50)

    stat = tk.Label(window, text='Detections:', bg='#003659' ,fg='white', font=("Samsung Sharp Sans", 10, "bold"), bd=0, justify='left')
    stat.place(anchor='c', x=200, y=400, width=220, height=20)

    stat = tk.Label(window, text='Scores:', bg='#003659' ,fg='white', font=("Samsung Sharp Sans", 10, "bold"), bd=0, justify='left')
    stat.place(anchor='c', x=200, y=500, width=220, height=20)

    

'''if __name__ == '__main__':
    while not rospy.is_shutdown():

        

        rate = rospy.Rate(10) # 10hz
        #NeuralNetwork()
        #rospy.logerr("Could not start MachineLearning_node")
        tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image_color", rosImage), Subscriber("/kinect2/sd/points", PointCloud2)], queue_size=1,slop=0.1)
        tss.registerCallback(collectImages)
        #try:

        global imageReturn
        global pointReturn

        with ky.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

            if spacePres == True:
                #tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image_color", rosImage), Subscriber("/kinect2/sd/points", PointCloud2)], queue_size=1,slop=0.1)
                #tss.registerCallback(detector_Callback)
                detector_Callback(imageReturn, pointReturn)
            #except:
            #rospy.logerr("failed to syncronize '/kinect2/hd/image' and '/kinect2/sd/points'")
        rate.sleep()
    #rospy.spin()'''

if __name__ == '__main__':
    gui()

    threads = App(window)


    window.title('Detector GUI')
    window.geometry("1920x1080")
    window.update()
    window.mainloop()
