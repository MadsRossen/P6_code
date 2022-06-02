#!/usr/bin/env python2

from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from point_cloud.msg import graspingCoor, planePoints
import numpy as np
import rospy
import ros_numpy
import math
import random 


# Callback function 
def pointcloudCallback(grasp, points):    
    pub = rospy.Publisher('points/graspingPoints', planePoints, queue_size=1)
    x0 = grasp.GraspingCoor.x
    y0 = grasp.GraspingCoor.y
    xref = grasp.ReferenceCoor.x
    yref = grasp.ReferenceCoor.y

    #defines how large an area to forward
    searchSpace = 10
    size = len(range(-searchSpace, searchSpace)) * len(range(-searchSpace, searchSpace))
    set = []  
    data = ros_numpy.numpify(points)
    #rospy.logerr(data[31][290])
    #rospy.logerr("length of points kinect")
    #rospy.logerr(data[100][100])
    # for loop picking points in a square around (x0,y0)
    for i in range(-searchSpace, searchSpace):
        for j in range (-searchSpace, searchSpace):
            if   math.isnan(data[y0+i][x0+j][0]) or math.isnan(data[y0+i][x0+j][1]) or math.isnan(data[y0+i][x0+j][2]):
                #rospy.logerr("remove")
                #rospy.logerr(data[y0+i][x0+j][0])
                pass
            else:
                set.append(Point(x = data[y0+i][x0+j][0], y = data[y0+i][x0+j][1], z = data[y0+i][x0+j][2]))
                
    # length of points published on topic
    len_of_data = 99
    #rospy.logerr(len(set))
    # random list of unique number 
    try:
        ran_list = random.sample(range(1, len(set)), len_of_data)
        pointData = []
        for i in range (len(ran_list)):
                pointData.append(set[ran_list[i]])
        msg = planePoints()
        msg.GraspingPoint = Point(x = data[y0][x0][0], y = data[y0][x0][1], z = data[y0][x0][2])
        msg.ReferencePoint = Point(x = data[yref][xref][0], y = data[yref][xref][1], z = data[yref][xref][2])
        msg.PlanePoints = pointData
        pub.publish(msg)
    except:
        rospy.logerr("Shit list is too short")

def main():
    rospy.init_node('point_cloud_grabber', anonymous=False)

    tss = ApproximateTimeSynchronizer([Subscriber("/rgb/graspingPoint", graspingCoor), Subscriber("/kinect2/sd/points", PointCloud2)],queue_size=1, slop=2, allow_headerless=True)
    tss.registerCallback(pointcloudCallback)
    # Simon test rosbag topic -> /camera/rgb/points
    rospy.spin()

if __name__ ==  '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

