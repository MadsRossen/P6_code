#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Point #Point is for testing without planePoints msg
from plane_creator.msg import TransRot

from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
#from tf.transformations import quaternion_matrix

import math
import ros_numpy

from scipy.spatial.transform import Rotation

import numpy as np

from point_cloud.msg import planePoints

class PointstoPlane():
    def __init__(self):
        # rospy.Subscriber("/kinect2/sd/points", PointCloud2, self.ConvertCallback)
        rospy.Subscriber("points/graspingPoints", planePoints, self.ConvertCallback) # This will be the actual topic and message as far as i can tell

    ##Converts the data from message frompoints to something we can work with
    def ConvertCallback(self, msg):
        pointListX = []
        pointListY = []
        pointListZ = []

        ## points for testing
        # p1 = Point(x=1, y=2, z=3)
        # p2 = Point(x=2, y=5, z=4)
        # p3 = Point(x=6, y=4, z=2)
        # p4 = Point(x=-2.34, y=4.09, z=5.25)
        # p5 = Point(x=-2.38, y=0.77, z=3.73)
        # p6 = Point(x=4.51, y=8.45, z=4.63)
        # p7 = Point(x=1.19, y=-1.17, z=1.47)
        # self.grasp = Point(x=0.69, y=2.53, z=3.36)
        # self.ref = Point(x=-7.01, y=9.46, z=9.52)
        # DataPoints = [p1, p2, p3, p4, p5, p6, p7]

        ## Actual points
        DataPoints = msg.PlanePoints
        self.grasp = msg.GraspingPoint
        self.ref = msg.ReferencePoint

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

        self.MakePointList(pointListX, pointListY, pointListZ)

    ##Makes a list of sk.spatial points based on the geometry_msgs points 
    def MakePointList(self, pointListX, pointListY, pointListZ):
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
        self.CreatePlane(pointList)

    ##Creates a plane based on the sk.spatial points
    def CreatePlane(self, pointList):
        plane = Plane.best_fit(pointList)

        self.PlanePointAndNormal(plane)

    ## split the plane to the centroid point and normal vector
    def PlanePointAndNormal(self, plane):
        centroid = []
        centroid.append(plane.point[0])
        centroid.append(plane.point[1])
        centroid.append(plane.point[2])

        normvector = []
        normvector.append(plane.normal[0])
        normvector.append(plane.normal[1])
        normvector.append(plane.normal[2])

        self.ConvertToPose(centroid, normvector)

    ## Converts centroid and normal vector to geometry_msg/Pose (point and quaternions)
    def ConvertToPose(self, centroid, normvector):
        msg = TransRot()
        msg.translationVector[0] = (self.grasp.x * -1) - 0.085 
        msg.translationVector[1] = self.grasp.y + 0.52248
        msg.translationVector[2] = 1.0271 - (self.grasp.z + 0.008 - 0.1)

        #rospy.logerr(1.0271-(self.grasp.z + 0.008))

        xVectorCam = [1, 0, 0]
        yVectorCam = [0, 1, 0]
        zVectorCam = [0, 0, 1]

        xPunkt = self.ref

        xVector = [centroid[0]+xPunkt.x, centroid[1]+xPunkt.y, centroid[2]+xPunkt.z]
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

        self.PublishPlane(msg)

    ## Publishes the grasping pose for the plane in relation to the camera
    def PublishPlane(self, msg):
        pub = rospy.Publisher('/grasping_pose', TransRot, queue_size=1)
        rate = rospy.Rate(10) # 10hz
        pub.publish(msg)
        rate.sleep()

def main():
    rospy.init_node('plane_creator', anonymous=False)
    PointstoPlane()
    rospy.spin()
    
if __name__ ==  '__main__':
    main()