#!/usr/bin/env python
import rospy
from std_msgs.msg import Header

    

if __name__ ==  '__main__':
    rospy.init_node('trigger_node', anonymous=False)
    pub = rospy.Publisher('/trigger', Header, queue_size=1)
    msg = Header()
    msg.stamp = rospy.Time.now()
    pub.publish(msg)