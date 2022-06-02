from std_msgs.msg import Header

try:
        tss = ApproximateTimeSynchronizer(
            [Subscriber("/kinect2/hd/image", Image), Subscriber("/kinect2/sd/points", PointCloud2), Subscriber("/tigger", Header)], queue_size=1,
            slop=2)
        tss.registerCallback(detector_Callback)
    except:
