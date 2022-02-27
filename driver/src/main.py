from math import degrees
import sys
import rospy
from std_msgs.msg import Bool
import time
from acf_network.msg import SceneObject, SceneObjectArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

import numpy as np
from scipy.spatial.transform import Rotation as R


class StateMachine():
    def __init__(self):
        self.next_state = "wait_for_user"
        self.run_net_pub = rospy.Publisher("/ACF_Network/run_network", Bool, queue_size=1)
        self.detection_sub = rospy.Subscriber("/ACF_Network/detections/scene_objects", SceneObjectArray, self.received_detections)
        self.grasp_pub = rospy.Publisher("/move_group/grasp_pose", PoseStamped, queue_size=1)
        self.pour_pub = rospy.Publisher("/move_group/pour_action_poses", PoseArray, queue_size=1)
        self.print = True
        self.obj_map = {}
        self.scene_object_pub = rospy.Publisher("/move_group/scene_objects", SceneObjectArray, queue_size=1)

    def run(self):
        if self.next_state == "wait_for_user":
            self.wait_for_user_input()
        elif self.next_state == "wait_for_detections":
            self.wait_for_detections()
            self.print = False
        elif self.next_state == "process_detections":
            self.process_detections()
        elif self.next_state == "calculate_grasp_pose":
            self.calculate_grasp_pose()
        elif self.next_state == "wait_for_grasp":
            self.wait_for_grasp()
    
    def wait_for_user_input(self):
        ans = input("Run network? (y/n): ")
        if ans == 'y':
            self.run_net_pub.publish(True)
            self.next_state = "wait_for_detections"
            self.print = True
        elif ans == 'q':
            print("Quitting..")
            sys.exit(0)
    
    def wait_for_detections(self):
        if self.print:
            print("Waiting for the network to return detections")
    
    def received_detections(self, msg):
        print("Received detections!")
        self.obj_map.clear()
        self.detections = msg
        for i, obj in enumerate(self.detections.scene_objects):
            self.obj_map[i] = obj
            print("{}: {}".format(i, obj.name))
        self.next_state = "process_detections"

    def process_detections(self):
        choice = str(input("What action would you like to take? Grasp (g), Pour (p) or Stir (s) "))
        if choice == 'g' or choice == "G":
            self.grasp()
        elif choice == 'p' or choice == 'P':
            self.pour()
        elif choice == 's' or choice == 'S':
            self.stir()
        else:
            print("Not a valid choice. Quitting. . .")
            sys.exit(0)
        
    
    def grasp(self):
        choice = int(input("Which object would you like to grasp? "))
        self.obj_to_grasp = self.obj_map[choice]
        print("User selection: {}".format(self.obj_to_grasp.name))
        grasp_pose_stamped = self.calculate_grasp_pose()
        grasp_pose_stamped.pose.orientation.w = 1
        print("Grasp pose calculated. position = ({}, {}, {}) and orientation = ({}, {}, {})".format(grasp_pose_stamped.pose.position.x, grasp_pose_stamped.pose.position.y, grasp_pose_stamped.pose.position.z, grasp_pose_stamped.pose.orientation.x, grasp_pose_stamped.pose.orientation.y, grasp_pose_stamped.pose.orientation.z, grasp_pose_stamped.pose.orientation.w))
        scene_objs_arr = SceneObjectArray()
        scene_objs_arr.scene_objects.append(self.obj_to_grasp)
        self.scene_object_pub.publish(scene_objs_arr)
        rospy.sleep(2)
        self.grasp_pub.publish(grasp_pose_stamped)
        self.next_step = "wait_for_grasp"
    
    def pour(self):
        """
            Pouring object 
            Container object
        """
        pour_choice = int(input("Which object would you like to be the pouring object? "))
        contain_choice = int(input("Which object would you like to be the contain object? "))
        pour_obj = self.obj_map[pour_choice]
        contain_obj = self.obj_map[contain_choice]
        self.obj_to_grasp = pour_obj
        pour_obj_grasp_pose_stamped = self.calculate_grasp_pose()
        pour_obj_grasp_pose = pour_obj_grasp_pose_stamped.pose
        self.obj_to_grasp = contain_obj
        contain_obj_grasp_pose_stamped = self.calculate_grasp_pose()
        contain_obj_grasp_pose = contain_obj_grasp_pose_stamped.pose
        pose_arr_msg = PoseArray()
        pose_arr_msg.header.frame_id = "/base_link"
        pose_arr_msg.poses.append(pour_obj_grasp_pose)
        pose_arr_msg.poses.append(contain_obj_grasp_pose)
        self.pour_pub.publish(pose_arr_msg)

    
    def stir(self):
        stir_choice = int(input("Which object would you like to be the stirring object? "))
        stir_obj = self.obj_map[stir_choice]
        stir_obj_grasp_pose = self.calculate_grasp_pose(stir_obj)
    
    def calculate_grasp_pose(self):
        print("Calculating grasp pose for {}".format(self.obj_to_grasp.name))


        if self.obj_to_grasp.name.startswith("OC"):
            part_acf = self.obj_to_grasp.kp1.pose
            part_acf_position = np.array([part_acf.position.x, part_acf.position.y, part_acf.position.z]).reshape(3,1)
            grasp_y_axis = np.array([0, 0, 1]).reshape(1,3) #euler_angles
            grasp_z_axis = np.cross(grasp_y_axis, np.random.rand(3, ))
            grasp_z_axis = grasp_z_axis / np.linalg.norm(grasp_z_axis)
            grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
            grasp_pose_mat = np.vstack((grasp_x_axis, grasp_y_axis, grasp_z_axis)).T
            r = R.from_matrix(grasp_pose_mat)
            grasp_pose_quat = r.as_quat()
            grasp_pose_stamped = PoseStamped()
            grasp_pose_stamped.header.frame_id = '/base_link'
            grasp_pose_stamped.header.stamp = rospy.Time.now()
            grasp_pose_stamped.pose.position.x = part_acf_position[0][0]
            grasp_pose_stamped.pose.position.y = part_acf_position[1][0]
            grasp_pose_stamped.pose.position.z = part_acf_position[2][0]
            rospy.loginfo("grasp_pose positions ({:.4f}, {:.4f}, {:.4f})".format(grasp_pose_stamped.pose.position.x, grasp_pose_stamped.pose.position.y, grasp_pose_stamped.pose.position.z))
            grasp_pose_stamped.pose.orientation.x = grasp_pose_quat[0]
            grasp_pose_stamped.pose.orientation.y = grasp_pose_quat[1]
            grasp_pose_stamped.pose.orientation.z = -grasp_pose_quat[2]
            grasp_pose_stamped.pose.orientation.w = grasp_pose_quat[3]

        elif self.obj_to_grasp.name.startswith("mug"):
            print(self.obj_to_grasp)
            body_acf = self.obj_to_grasp.kp1.pose
            handle_acf = self.obj_to_grasp.kp2.pose
            handle_axis = R.from_quat([handle_acf.orientation.x, handle_acf.orientation.y, handle_acf.orientation.z, handle_acf.orientation.w])
            handle_axis = handle_axis.as_euler('xyz', degrees=False)
            handle_axis_list = [handle_axis[i] for i in range(3)]
            body_axis = R.from_quat([body_acf.orientation.x, body_acf.orientation.y, body_acf.orientation.z, handle_acf.orientation.w])
            body_axis = body_axis.as_euler('xyz', degrees=False)
            body_axis_list = [body_axis[i] for i in range(3)]


            grasp_z_axis = np.cross(handle_axis, body_axis)
            grasp_z_axis = grasp_z_axis / np.linalg.norm(grasp_z_axis)
            grasp_y_axis = np.cross(grasp_z_axis, body_axis) # offset axis
            grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
            grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
            grasp_pose_mat = np.vstack((grasp_x_axis, grasp_y_axis, grasp_z_axis)).T
            r = R.from_matrix(grasp_pose_mat)
            grasp_pose_quat = r.as_quat()
            grasp_pose_stamped = PoseStamped()
            grasp_pose_stamped.header.frame_id = '/base_link'
            grasp_pose_stamped.header.stamp = rospy.Time.now()
            # calculate handle offset
            handle_offset_const = 0.1 # 10 cm
            handle_offset_vector = handle_offset_const * grasp_y_axis
            grasp_pose_stamped.pose.position.x = handle_acf.position.x #- handle_offset_vector[0]
            grasp_pose_stamped.pose.position.y = handle_acf.position.y #- 0.1#- handle_offset_vector[1]
            grasp_pose_stamped.pose.position.z = handle_acf.position.z #- handle_offset_vector[2]
            rospy.loginfo("grasp_pose positions ({:.4f}, {:.4f}, {:.4f})".format(grasp_pose_stamped.pose.position.x, grasp_pose_stamped.pose.position.y, grasp_pose_stamped.pose.position.z))
            

        return grasp_pose_stamped
    
    def wait_for_grasp(self):
        if self.print:
            print("Wait for fetch to grasp the object")
            self.print = False
        self.next_state = "wait_for_user"

    
        


            


if __name__ == '__main__':
    rospy.init_node("driver_node")
    foo = StateMachine()
    while not rospy.is_shutdown():
        foo.run()

