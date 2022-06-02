#!/usr/bin/env python2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from point_cloud.msg import graspingCoor, pixelCoor
import rospy



def Callback(msg2):
    pub = rospy.Publisher('/rgb/graspingPoint', graspingCoor, queue_size=1)
    
    rate = rospy.Rate(10) # 10hz
 

    msg = graspingCoor()
    msg.header.stamp = msg2.header.stamp
    msg.header.frame_id = "navigation"
    msg.header.seq = msg2.header.seq
    msg.GraspingCoor  = pixelCoor(y = 190, x = 226) # grasping pixel fram MachineLearning
    msg.ReferenceCoor = pixelCoor(y = 160, x = 226) # reference pixel/point for transformation
    pub.publish(msg)
    rospy.loginfo(msg)
    rate.sleep()

def main():
    rospy.init_node('MachineLearning_node', anonymous=True)
    rospy.Subscriber("/kinect2/sd/points", PointCloud2, Callback)
    rospy.spin()

    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

