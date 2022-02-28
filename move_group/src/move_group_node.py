#!/usr/bin/python2

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from trajectory_msgs.msg import JointTrajectoryPoint
from acf_network.msg import SceneObject, SceneObjectArray
import tf



class MoveGroup():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.fetch = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_goal_position_tolerance(.03)
        self.move_group.set_goal_orientation_tolerance(.02)
        self.move_group.clear_pose_targets()
        self.display_traj_pub = rospy.Publisher("/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.grasp_sub = rospy.Subscriber("/move_group/grasp_pose", geometry_msgs.msg.PoseStamped, self.grasp_cb)
        self.pour_sub = rospy.Subscriber("/move_group/pour_action_poses", geometry_msgs.msg.PoseArray, self.pour_cb)
        # self.scene_object_sub = rospy.Subscriber("/move_group/scene_objects", SceneObjectArray, self.add_scene_objects)
        self.waypoints = []
        self.low_rectangle_table_height = 0.62
        self.high_dark_circle_table_height = 0.75
        # Give things time to initialize
        rospy.sleep(2)
        self.add_table()

    def pour_cb(self, msg):
        rospy.logwarn("Got pour msg: {}".format(msg))
        pour_obj_pose = msg.poses[0]
        contain_obj_pose = msg.poses[1]
        pour_obj_pose_stamped = geometry_msgs.msg.PoseStamped()
        pour_obj_pose_stamped.header.frame_id = "/base_link"
        pour_obj_pose_stamped.pose = pour_obj_pose

        # pour_obj_pose_stamped.pose.position.x -= 0.15
        contain_obj_pose_stamped = geometry_msgs.msg.PoseStamped()
        contain_obj_pose_stamped.header.frame_id = "/base_link"
        contain_obj_pose_stamped.pose = contain_obj_pose

        self.add_object_to_scene('pour_obj', pour_obj_pose_stamped, (.08, .08, .23))
        self.add_object_to_scene('contain_obj', contain_obj_pose_stamped, (.1, .1, .08))

        # first grasp the pour object
        grasps = self.generate_grasps(pour_obj_pose_stamped, obj_type="bottle")
        rospy.sleep(2)
        self.fetch.arm.pick('pour_obj', grasps)
        rospy.sleep(5) # sleep for 5 s to allow arm to grasp pour_obj... there is no wait=True argument for pick()




        prepour_pose_stamped = geometry_msgs.msg.PoseStamped()
        prepour_pose_stamped.header.frame_id = "/base_link"
        prepour_pose_stamped.pose = contain_obj_pose_stamped.pose
        prepour_pose_stamped.pose.position.z += 0.20
        prepour_pose_stamped.pose.position.x -= 0.20
        prepour_pose_stamped.pose.position.y -= 0.17
        prepour_pose_stamped.pose.orientation.x = 0
        prepour_pose_stamped.pose.orientation.y = 0
        prepour_pose_stamped.pose.orientation.z = 0
        prepour_pose_stamped.pose.orientation.w = 1
        rospy.loginfo("Setting pose target to {}".format(prepour_pose_stamped))
        self.move_group.set_pose_target(prepour_pose_stamped)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        pour_pose_stamped = copy.deepcopy(prepour_pose_stamped)
        pour_pose_stamped.pose.orientation.x = -0.7071068
        pour_pose_stamped.pose.orientation.w = 0.7071068
        self.move_group.set_pose_target(pour_pose_stamped)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        
    def grasp_cb(self, msg):
        # add object to be grasped to scene
        self.add_object_to_scene('grasp_obj', msg, (0.0254, 0.0254, .2))
        # rospy.sleep(2)
        rospy.loginfo(msg)
        grasps = self.generate_grasps(msg, obj_type="bottle")
        # rospy.loginfo(grasps)     

        rospy.sleep(2)
        self.fetch.arm.pick('grasp_obj', grasps)


    def generate_grasps(self, pose_stamped, obj_type):
        grasps = []
        g_x = moveit_msgs.msg.Grasp()
        g_x.id = "pick_along_x"
        grasp_pose = pose_stamped
        # grasp_pose.pose.position.x -= 0.17 # moveit plans in wrist_roll_link subtract 17 for ee link
        if obj_type == "bottle":
            grasp_pose.pose.position.x -= 0.13
            grasp_pose.pose.position.z += 0.02
            grasp_pose.pose.orientation.x = 0
            grasp_pose.pose.orientation.y = 0
            grasp_pose.pose.orientation.z = 0
            grasp_pose.pose.orientation.w = 1
        elif obj_type == "mug":
            grasp_pose.pose.position.x -= .12
            grasp_pose.pose.position.y -= 0.02

        # Always try grasp along x direction in base link
        # set grasp_pose
        g_x.grasp_pose = grasp_pose

        # define pre-grasp approach
        g_x.pre_grasp_approach.direction.header.frame_id = 'gripper_link'
        g_x.pre_grasp_approach.direction.vector.x = 1.0
        g_x.pre_grasp_approach.min_distance = 0.15 # m
        g_x.pre_grasp_approach.desired_distance = 0.17 # m

        # set post-grasp retreat up
        g_x.post_grasp_retreat.direction.header.frame_id = 'base_link' 
        g_x.post_grasp_retreat.direction.vector.z = 1.0
        g_x.post_grasp_retreat.desired_distance = 0.25
        g_x.post_grasp_retreat.min_distance = 0.2

        # set pre-grasp posture
        g_x.pre_grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
        pos = JointTrajectoryPoint()
        if obj_type =="mug":
            pos.positions.append(0.015) # mug handles are small. close grippers a little so we don't hit the body with the fingers
            pos.positions.append(0.015)
            # pos.time_from_start = rospy.Duration(0.5)
        else:
            pos.positions.append(0.5)
            pos.positions.append(0.5)
            pos.time_from_start = rospy.Duration(3)
        g_x.pre_grasp_posture.points.append(pos)

        # set grasp posture
        g_x.grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
        pos = JointTrajectoryPoint()
        pos.positions.append(0.0) # close grippers
        pos.positions.append(0.0)
        pos.effort.append(0.0)
        pos.effort.append(0.0)
        g_x.grasp_posture.points.append(pos)

        g_x.allowed_touch_objects = ['grasp_obj', '<octomap>', 'r_gripper_link', 'l_gripper_link','gripper_link']

        g_x.max_contact_force = 0
        grasps.append(g_x)
        # if obj_type == "mug":
        #     # try from y and minus y as well because it's easier for fetch
        #     g_y = moveit_msgs.msg.Grasp()
        #     g_y.id = "pick_along_y"
        #     grasp_pose = pose_stamped
        #     grasp_pose.pose.orientation.z = 0.7071068
        #     grasp_pose.pose.orientation.w =  0.7071068
        #     g_y.grasp_pose = grasp_pose

        #     # define pre-grasp approach
        #     g_y.pre_grasp_approach.direction.header.frame_id = 'gripper_link'
        #     g_y.pre_grasp_approach.direction.vector.z = -1.0
        #     g_y.pre_grasp_approach.min_distance = 0.1 # 1
        #     g_y.pre_grasp_approach.desired_distance = 0.15 # m

        #     # set post-grasp retreat
        #     g_y.post_grasp_retreat.direction.header.frame_id = 'base_link' 
        #     g_y.post_grasp_retreat.direction.vector.z = 1.0
        #     g_y.post_grasp_retreat.desired_distance = 0.15
        #     g_y.post_grasp_retreat.min_distance = 0.1

        #     # set pre-grasp posture
        #     g_y.pre_grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
        #     pos = JointTrajectoryPoint()
        #     pos.positions.append(0.05) # open fingers all the way
        #     pos.positions.append(0.05)
        #     g_y.pre_grasp_posture.points.append(pos)

        #     # set grasp posture
        #     g_y.grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
        #     pos = JointTrajectoryPoint()
        #     pos.positions.append(0.0) # close the grippers
        #     pos.positions.append(0.0)
        #     pos.effort.append(0.0)
        #     pos.effort.append(0.0)
        #     g_y.grasp_posture.points.append(pos)

        #     g_y.allowed_touch_objects = ['grasp_obj', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'gripper_link']

        #     g_y.max_contact_force = 0
        #     grasps.append(g_y)

            # g_minusy = moveit_msgs.msg.Grasp()
            # g_minusy.id = "pick_along_minus_y"
            # grasp_pose = pose_stamped
            # g_minusy.grasp_pose = grasp_pose

            # # define pre-grasp approach
            # g_minusy.pre_grasp_approach.direction.header.frame_id = 'gripper_link'
            # g_minusy.pre_grasp_approach.direction.vector.y = -1.0
            # g_minusy.pre_grasp_approach.min_distance = 0.2 # 1
            # g_minusy.pre_grasp_approach.desired_distance = 0.25 # m

            # # set post-grasp retreat
            # g_minusy.post_grasp_retreat.direction.header.frame_id = 'base_link' 
            # g_minusy.post_grasp_retreat.direction.vector.z = 1.0
            # g_minusy.post_grasp_retreat.desired_distance = 0.15
            # g_minusy.post_grasp_retreat.min_distance = 0.1

            # # set pre-grasp posture
            # g_y.pre_grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
            # pos = JointTrajectoryPoint()
            # pos.positions.append(0.05) # open fingers all the way
            # pos.positions.append(0.05)
            # g_y.pre_grasp_posture.points.append(pos)
            # rospy.loginfo(pos)

            # # set grasp posture
            # g_y.grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
            # pos = JointTrajectoryPoint()
            # pos.positions.append(0.0) # close the grippers
            # pos.positions.append(0.0)
            # pos.effort.append(0.0)
            # pos.effort.append(0.0)
            # g_y.grasp_posture.points.append(pos)

            # g_y.allowed_touch_objects = ['grasp_obj', 'r_gripper_finger_joint', 'l_gripper_finger_joint']

            # g_y.max_contact_force = 0
            # grasps.append(g_y)

        if obj_type =="bottle":
            rospy.logwarn("obj is a bottle")
            g_z = moveit_msgs.msg.Grasp()
            g_z.id = "pick_along_z"
            grasp_pose = pose_stamped
            grasp_pose.pose.position.x -=0.07
            # set grasp_pose
            g_z.grasp_pose = grasp_pose

            # define pre-grasp approach
            g_z.pre_grasp_approach.direction.header.frame_id = 'gripper_link'
            g_z.pre_grasp_approach.direction.vector.z = -1.0
            g_z.pre_grasp_approach.min_distance = 0.15 # m
            g_z.pre_grasp_approach.desired_distance = 0.17 # m

            # set post-grasp retreat
            g_z.post_grasp_retreat.direction.header.frame_id = 'base_link' 
            g_z.post_grasp_retreat.direction.vector.z = 1.0
            g_z.post_grasp_retreat.desired_distance = 0.25
            g_z.post_grasp_retreat.min_distance = 0.2

            # set pre-grasp posture
            g_z.pre_grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
            pos = JointTrajectoryPoint()
            pos.positions.append(0.05) 
            pos.positions.append(0.05)
            g_z.pre_grasp_posture.points.append(pos)

            # set grasp posture
            g_z.grasp_posture.joint_names = ['r_gripper_finger_joint', 'l_gripper_finger_joint']
            pos = JointTrajectoryPoint()
            pos.positions.append(0.0)
            pos.positions.append(0.0)
            pos.effort.append(0.0)
            pos.effort.append(0.0)
            g_z.grasp_posture.points.append(pos)

            
            g_z.allowed_touch_objects = ['grasp_obj', '<octomap>', 'r_gripper_link', 'l_gripper_link','gripper_link']

            g_z.max_contact_force = 0
            grasps.append(g_z)
        return grasps

    def add_table(self):

        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = '/base_link'
        table_pose.pose.position.x = .9
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = self.low_rectangle_table_height/2
        table_pose.pose.orientation.x = 0
        table_pose.pose.orientation.y = 0
        table_pose.pose.orientation.z = 0
        table_pose.pose.orientation.w = 1
        # high dark circle table
        self.add_object_to_scene('table', table_pose, (1, 1.3, self.low_rectangle_table_height))
        # low rectangle table 
        # self.add_object_to_scene('table', table_pose, (0.65, 1.3, self.table_height))
        self.move_group.set_support_surface_name('table')

    # def add_scene_objects(self, msg):
    #     for obj in msg.scene_objects:
    #         obj_body_pose = obj.kp1
    #         # if obj.name.startswith("OC"):
    #         #     obj_size = (0.05, 0.05, 0.1)
    #         # elif obj.name.startswith("mug"):
    #         #     object_size = (0, 0, 0)
    #         object_size = (0.05, 0.05, 0.1)
    #         self.add_object_to_scene(obj.name, obj_body_pose, object_size)

    def add_object_to_scene(self, object_name, object_pose, object_size):
        object_pose.pose.orientation.x = 0
        object_pose.pose.orientation.y = 0
        object_pose.pose.orientation.z = 0
        object_pose.pose.orientation.w = 1
        if object_name.endswith("_obj"):
            # this is an object that goes on the table
            pose_with_buffer = copy.deepcopy(object_pose)
            pose_with_buffer.pose.position.z = object_size[2]/2 + self.low_rectangle_table_height + 0.003 # 0.003 is some buffer so moveit does not complain about collision between objects in scene
            self.scene.add_box(object_name, pose_with_buffer, object_size)        
        else:
            self.scene.add_box(object_name, object_pose, object_size)


if __name__ == '__main__':
    rospy.init_node('py_move_group_node', anonymous=True)
    foo = MoveGroup()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()

