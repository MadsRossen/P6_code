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
    rospy.logerr("inside callback")
    
    
        #try:
    msg = graspingCoor()
    msg.header.stamp = pointcloud.header.stamp
    msg.header.frame_id = "graspingCoordinate"
    msg.header.seq = pointcloud.header.seq
    #msg.GraspingCoor = pixelCoor(x=cX, y=cY)  # grasping pixel fram MachineLearning
    #msg.ReferenceCoor = pixelCoor(x=rX, y=rY)  # reference pixel/point for transformation
    msg.GraspingCoor = pixelCoor(x=860, y=480)
    msg.ReferenceCoor = pixelCoor(x=860, y=440)
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
    len_of_data= len(set)


    rospy.logerr(len(set))
    # random list of unique number 
    #try:
    ran_list = random.sample(range(0, len(set)), len_of_data)
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
    rospy.logwarn(msg)
    rospy.logerr("Grasping pose published")
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
