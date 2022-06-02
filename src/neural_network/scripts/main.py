#!/usr/bin/env python3
from sensor_msgs.msg import PointCloud2, Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from point_cloud.msg import graspingCoor, pixelCoor
from std_msgs.msg import Bool
import rospy
import numpy as np
import os
import sys
import numpy as np
import cv2
import imutils
import tensorflow as tf
from mrcnn.config import Config
import mrcnn.model as modellib

try:
    # Root directory of the project
    ROOT_DIR = os.path.abspath("")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    WEIGHTS_PATH = ("/home/jarvis/P6_project/src/neural_network/pretrained_weightsmask_rcnn_object_0004.h5")

    class CustomConfig(Config):
        NAME = "object"
        IMAGES_PER_GPU = 2
        NUM_CLASSES = 1 + 1
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = 0.98

    config = CustomConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7


    config = InferenceConfig()
    config.display()

    DEVICE = "/cpu:0"
    TEST_MODE = "inference"
except:
    rospy.logerr("could not initialize Mask R-CNN")



def detector_Callback(image, pointcloud):
    pub = rospy.Publisher('/maskRCNN/graspingPoint', graspingCoor, queue_size=1)
    pub_pointcloud = rospy.Publisher('/maskRCNN/pointCloud', PointCloud2, queue_size=1)
    
    try:
        cv2_img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)


        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=config)
        weights_path = WEIGHTS_PATH
        model.load_weights(weights_path, by_name=True)

        results1 = model.detect([cv2_img], verbose=1)

        r1 = results1[0]

        scores = r1['scores']
        mask = r1['masks']
        filtered = filter(lambda num: num > 0.98, scores)


        if mask.shape[2] != 0:
            for i in range(len(list(filtered))):
                if i == 1:
                    break

                img = mask[:,:,i].astype('uint8')
                img *= 255
                edged = cv2.Canny(img, 30, 200)
                contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(contours)
                c = max(cnts, key=cv2.contourArea)

                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
    except:
        rospy.logerr("Could not calculate masks")
 
    try:
        msg = graspingCoor()
        msg.header.stamp = pointcloud.header.stamp
        msg.header.frame_id = "graspingCoordinate"
        msg.header.seq = pointcloud.header.seq
        msg.GraspingCoor  = pixelCoor(x = cX, y = cY) # grasping pixel fram MachineLearning
        msg.ReferenceCoor = pixelCoor(x = 200, y = 200) # reference pixel/point for transformation
        pub.publish(msg) # grasping coordinate
        pub_pointcloud.publish(pointcloud) # pointcloud matching the image detected on
    except:
        rospy.logerr("Cannot publish grasping coordinate")

def trigger_callback(msg):
    try:
        tss = ApproximateTimeSynchronizer([Subscriber("/kinect2/hd/image", Image), Subscriber("/kinect2/sd/points", PointCloud2)],queue_size=10, slop=0.1)
        tss.registerCallback(detector_Callback)
    except:
        rospy.logerr("failed to syncronize '/kinect2/hd/image' and '/kinect2/sd/points'")




def main():
    rospy.init_node('MachineLearning_node', anonymous=True)
    rospy.Subscriber("/trigger", Bool, trigger_callback)
    rospy.spin()

    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("Could not start MachineLearning_node")
