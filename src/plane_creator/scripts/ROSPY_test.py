import rospy

from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt

pointList = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])

print('Pointlist: ', type(pointList))
print(type(pointList[0]))
print(type(pointList[0][1]))

def CreatePlane(pointList):
    plane = Plane.best_fit(pointList)

    xyArray = []
    for x in range(len(pointList)):
        xyArray.append(pointList[x][0])
        xyArray.append(pointList[x][1])

    limsMax = max(xyArray)
    limsMin = min(xyArray)

    point = plane.point

    plot_3d(pointList.plotter(c='k', s=50, depthshade=False), plane.plotter(alpha=0.2, lims_x=(limsMin, limsMax), lims_y=(limsMin, limsMax)), point.plotter(c='r', s=50, depthshade=False))

    rospy.Subscriber("/kinect2/sd/points ", kinect2/sd/points , callback)
 
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

#    plt.show()

CreatePlane(pointList)