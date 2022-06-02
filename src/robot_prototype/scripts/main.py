#!/usr/bin/env python2

from os import stat
import sys
import copy
from time import sleep
from matplotlib.cbook import to_filehandle

from numpy import PINF, uint8 
import rospy
import moveit_commander
import moveit_msgs.msg
from ur_msgs.msg import Digital
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String, Bool
from ur_msgs.srv import SetIO
from moveit_commander.conversions import pose_to_list
from tf.transformations import *
from scipy.spatial.transform import Rotation
from plane_creator.msg import TransRot

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True





class droid(object):
    def __init__(self):
        super(droid, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        #moveit_commander.roscpp_initialize(sys.argv)

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_prototype', anonymous=False)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()


        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "robot"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        move_group.set_num_planning_attempts(100)

        move_group.set_planning_time(2)
        move_group.set_planner_id("RRTConnect")

        move_group.set_goal_tolerance(0.005)
        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_pose_goal(self, x,y,z,rx,ry,rz):
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # Planning to a Pose Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^
        # Plan a motion for this group to a desired pose for the
        # end-effector:
        
     
        '''pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = qx
        pose_goal.orientation.y = qy
        pose_goal.orientation.z = qz
        pose_goal.orientation.w = qw
        pose_goal.position.x = -x
        pose_goal.position.y = -y
        pose_goal.position.z = z'''

        pose_goal = [-x, -y, z, rx, ry, rz]
#        rospy.logerr("Desired goal")
#        rospy.logerr(pose_goal)
        moved = False
        count = 0
        tolerance = 0.005
        move_group.set_pose_target(pose_goal)
        thePlan = move_group.plan()
        while(moved == False):
            move_group.set_pose_target(pose_goal)
            thePlan = move_group.plan()
            moved = move_group.execute(thePlan, wait=True)
#            rospy.logwarn("execute:")
#            rospy.logwarn(moved)
            count = count + 1
            if moved == True or count == 3:
#                rospy.logerr("broken breaker!")
                break


        
        #move_group.go(wait=True)
        #moveit_msgs.msg.MoveItErrorCodes.FAILURE


        #go = move_group.go(joints=pose_goal, wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()

        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

    
    def static_scene(self):
        '''
        Add the stactic scene to Rviz 
        '''

        planning_frame = self.planning_frame
        scene = self.scene


        # Add the table
        pose_table = geometry_msgs.msg.PoseStamped()
        id_table = 'table'
        pose_table.header.frame_id = planning_frame
        pose_table.pose.position.x = 0
        pose_table.pose.position.y = 0
        pose_table.pose.position.z = 0
        scale_table = (0.001, 0.001, 0.001)
        mesh_path_table = "/home/jarvis/P6_project/src/robot_prototype/mesh/TableRobotCell.stl"

        scene.add_mesh(id_table, pose_table, mesh_path_table, scale_table)

        pose_cam = geometry_msgs.msg.PoseStamped()
        id_cam = 'camera'
        pose_cam.header.frame_id = planning_frame
        pose_cam.pose.position.x = 0
        pose_cam.pose.position.y = -0.52248
        pose_cam.pose.position.z = 1.0271
        scale_cam = (0.001, 0.001, 0.001)
        mesh_path_cam = "/home/jarvis/P6_project/src/robot_prototype/mesh/kinect2.stl"

        scene.add_mesh(id_cam, pose_cam, mesh_path_cam, scale_cam)

    def Callback(self, msg):
        #while (all([ entry != 0 for entry in self.move_group.get_current_state().velocity])):
        #    rospy.sleep()
        #robot.go_to_pose_goal(-0.487369800035,-0.107552340766, 0.433662294791,0.552923103477,0.833086116708,-0.000566859486762,0.000165216837586)
#        rospy.logerr("callback started")
#        rospy.logerr(msg)
        Rotation_matrix = [[msg.rotationMatrix[0], msg.rotationMatrix[1], msg.rotationMatrix[2], 0], [msg.rotationMatrix[3], msg.rotationMatrix[4], msg.rotationMatrix[5], 0], [msg.rotationMatrix[6], msg.rotationMatrix[7], msg.rotationMatrix[8], 0], [0, 0, 0, 1]]
        #base_to_camera = quaternion_matrix([0, 1, 0, 0])
        base_to_camera = euler_matrix(pi, 0, pi, 'sxyz')
        resultion_rot_matrix = numpy.dot(base_to_camera, Rotation_matrix)
        
    
        grasp_angles = euler_from_matrix(Rotation_matrix)
        euler = euler_from_matrix(resultion_rot_matrix)
#        rospy.logerr(grasp_angles)
        #rospy.logerr(base_to_camera2)

        robot.go_to_pose_goal(msg.translationVector[0], msg.translationVector[1], msg.translationVector[2] + 0.25, euler[0] , euler[1] , euler[2] ) # 10cm above parcel
        robot.go_to_pose_goal(msg.translationVector[0], msg.translationVector[1], msg.translationVector[2] + 0.0 , euler[0] , euler[1] , euler[2] ) # parcel
        self.grip(True)
        robot.go_to_pose_goal(msg.translationVector[0], msg.translationVector[1], msg.translationVector[2] + 0.25, euler[0] , euler[1] , euler[2] ) # 10cm above parcel

        robot.go_to_pose_goal(-0.47184, -0.12314, 0.43551, pi/2, pi, 0.0) # drop off coordinate
        
        self.grip(False)
        pub_trigger = rospy.Publisher('/trigger2', String, queue_size=1)
        trigger = String()
        trigger.data = str(rospy.Time.now().to_sec())
        pub_trigger.publish(trigger)

    def grip(self, state):
        rospy.wait_for_service('/ur_hardware_interface/set_io')
        set_io = rospy.ServiceProxy('/ur_hardware_interface/set_io',SetIO)
        set_io(fun = 1, pin = 0, state = state)
        rospy.loginfo("Pin = 0: " + str(state))    




    def main(self):
        print(" inside main")
        #robot.go_to_pose_goal(-0.47184, -0.12314, 0.43551, pi/2, pi, 0.0) # drop off coordinate
        #robot.go_to_pose_goal(-0.26, 0.495, 0.33000, 0.29322, -0.64294, 0.64584, -0.28891)
        #robot.go_to_pose_goal(0.47184, 0.12314, 0.43000, 0.29322, -0.64294, 0.64584, -0.28891)
        ##robot.go_to_pose_goal(-0.47184, -0.12314, 0.43000, 0.29322, -0.64294, 0.64584, -0.28891)
        rospy.Subscriber('/grasping_pose', TransRot, self.Callback)
        rospy.spin()
    

if __name__ ==  '__main__':
    robot = droid()
    robot.static_scene()
    robot.main()

