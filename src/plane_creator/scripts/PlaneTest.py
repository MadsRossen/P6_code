#!/usr/bin/env python3

######################################################
##  This file is for testing all kinds of sillyness ##
######################################################
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
import std_msgs.msg as std_msg

from geometry_msgs.msg import Pose, Point
#import geometry_msgs.msg as geometry_msg

from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
import math
import ros_numpy

from scipy.spatial.transform import Rotation

import numpy as np

import random
from plane_creator.msg import Num

pointList = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])



class PointstoPlane():
    def __init__(self):
        rospy.Subscriber("/kinect2/sd/points", PointCloud2, self.ConvertCallback)
        # pub = rospy.Publisher('Plane_data', Num, queue_size=10)

    def ConvertCallback(self, msg):
        pointListX = []
        pointListY = []
        pointListZ = []
        data = ros_numpy.numpify(msg)

        ## points for testing
        p1 = Point(x=1, y=2, z=3)
        p2 = Point(x=2, y=5, z=4)
        p3 = Point(x=6, y=4, z=2)
        p4 = Point(x=-2.34, y=4.09, z=5.25)
        p5 = Point(x=-2.38, y=0.77, z=3.73)
        p6 = Point(x=4.51, y=8.45, z=4.63)
        p7 = Point(x=1.19, y=-1.17, z=1.47)
        self.grasp = Point(x=0.69, y=2.53, z=3.36)
        self.ref = Point(x=-7.01, y=9.46, z=9.52)

        DataPoints = [p1, p2, p3, p4, p5, p6, p7]

        ListOfPoints = []
        for point in DataPoints:
            ListOfPoints.append(ros_numpy.numpify(point))

        for i in range(0,len(ListOfPoints)):
            if math.isnan(ListOfPoints[i][0]) != True:
                pointListX.append(ListOfPoints[i][0])
            if math.isnan(ListOfPoints[i][1]) != True:
                pointListY.append(ListOfPoints[i][1])
            if math.isnan(ListOfPoints[i][2]) != True:
                pointListZ.append(ListOfPoints[i][2])

        # maxPointNum = 50
        # if len(data) > maxPointNum:
        #     for i in range(0,maxPointNum):
        #         for j in range(0, maxPointNum):
        #             if math.isnan(data[i][j][0]) != True:
        #                 pointListX.append(data[i][j][0])
        #             if math.isnan(data[i][j][1]) != True:
        #                 pointListY.append(data[i][j][1])
        #             if math.isnan(data[i][j][2]) != True:
        #                 pointListZ.append(data[i][j][2])
        # elif len(data) <= maxPointNum:
        #     for rows in data:
        #         for cols in rows:
        #             if math.isnan(cols[0]) != True:
        #                 pointListX.append(cols[0])
        #             if math.isnan(cols[1]) != True:
        #                 pointListY.append(cols[1])
        #             if math.isnan(cols[2]) != True:
        #                 pointListZ.append(cols[2])

        self.MakePointList(pointListX, pointListY, pointListZ)

    def MakePointList(self, pointListX, pointListY, pointListZ):
        # tempPointList = []
        # for i in range(len(pointListX)):
        #     tempPoint = []
        #     for j in range(3):
        #         if j == 0:
        #             tempPoint.append(pointListX[i])
        #         if j == 1: 
        #             tempPoint.append(pointListY[i])
        #         if j == 2:
        #             tempPoint.append(pointListZ[i])
        #     tempPointList.append(tempPoint)
        # pointList = Points(tempPointList)
        # self.CreatePlane(pointList)
        


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


    def CreatePlane(self, pointList):
        plane = Plane.best_fit(pointList)

        # xyArray = []
        # for x in range(len(pointList)):
        #     xyArray.append(pointList[x][0])
        #     xyArray.append(pointList[x][1])

        # limsMax = max(xyArray)
        # limsMin = min(xyArray)

        # point = plane.point

        # print('Plane point: ', type(plane.point), plane.point)
        # print('Plane normal: ', type(plane.normal), plane.normal)

        # plot_3d(pointList.plotter(c='k', s=50, depthshade=False), plane.plotter(alpha=0.2, lims_x=(limsMin, limsMax), lims_y=(limsMin, limsMax)), point.plotter(c='r', s=50, depthshade=False))
        # plt.show()
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

        # self.PublishPlane(centroid, normvector)
        self.ConvertToPose(centroid, normvector)

    ## Converts centroid and normal vector to geometry_msg/Pose (point and quaternions)
    def ConvertToPose(self, centroid, normvector):
        msg = Pose()
        msg.position = self.grasp


        zVectorCam = [0, 0, 1]
        xVectorCam = [1, 0, 0]
        yVectorCam = [0, 1, 0]

        xPunkt = self.ref

        xVector = [centroid[0]+xPunkt.x, centroid[1]+xPunkt.y, centroid[2]+xPunkt.z]
        #centroid + xPunkt #halver punktets
#        xVectorLen = math.sqrt(xVector[0]*xVector[0]+xVector[1]*xVector[1]+xVector[2]*xVector[2])
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


        Rotation_matrix = [[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]]

        #Implement scipy.spatial.transform.Rotation
        r = Rotation.from_matrix(Rotation_matrix)
        quaternions = r.as_quat()

        print('Position: ', self.grasp)
        print('Quats: ', quaternions)

        msg.orientation.x = quaternions[0]
        msg.orientation.y = quaternions[1]
        msg.orientation.z = quaternions[2]
        msg.orientation.w = quaternions[3]
        self.PublishPlane(msg)

    ## Publishes centroid and normal vector to plane
    def PublishPlane(self, msg):
        pub = rospy.Publisher('grasping_pose', Pose, queue_size=10)
        rate = rospy.Rate(10) # 10hz
        pub.publish(msg)
        rate.sleep()


        # while not rospy.is_shutdown():
        #     pub.publish(centroid, normvector)
        #     rate.sleep()



def main():
    rospy.init_node('plane_creator', anonymous=False)
    PointstoPlane()
    rospy.spin()
    
if __name__ ==  '__main__':
    main()