import rospy
from std_msgs.msg import Bool
import time
from acf_network.msg import SceneObject, SceneObjectArray
from geometry_msgs.msg import PoseStamped


class StateMachine():
    def __init__(self):
        self.next_state = "wait_for_network"
        self.run_net_pub = rospy.Publisher("/ACF_Network/run_network", Bool, queue_size=1)
        self.detection_sub = rospy.Subscriber("/ACF_Network/detections/scene_objects", SceneObjectArray, self.received_detections)
        self.net_ready_sub = rospy.Subscriber("/ACF_Network/ready_for_detections", Bool, self.wait_for_user_input)
        self.grasp_pose_pub = rospy.Publisher("/grasp_pose", PoseStamped, queue_size=1)
        self.print = True

    def run(self):
        if self.next_state == "wait_for_network":
            self.wait_for_net()
        elif self.next_state == "wait_for_user":
            print("Next state is wait for detections")
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
    
    def wait_for_net(self):
        if self.print:
            print("Waiting for the network to initialize. . .")
            self.print = False
    
    def wait_for_user_input(self, msg):
        ans = input("Run network? (y/n): ")
        if ans == 'y':
            self.run_net_pub.publish(True)
            self.next_state = "wait_for_detections"
            self.print = True
    
    def wait_for_detections(self):
        if self.print:
            print("Waiting for the network to return detections")
    
    def received_detections(self, msg):
        self.next_state = "process_detections"
        self.detections = msg
    
    def process_detections(self):
        print("Received detections!")
        obj_map = {}
        for i, obj in enumerate(self.detections.scene_objects):
            obj_map[i] = obj
            print("\t{}: {}".format(i,obj.name))
        choice = int(input("Which object would you like to grasp? "))
        self.obj_to_grasp = obj_map[choice]
        print("User selection: {}".format(self.obj_to_grasp.name))
        self.next_state = "calculate_grasp_pose"
    
    def calculate_grasp_pose(self):
        grasp_pose = self.obj_to_grasp.kp1
        self.grasp_pose_pub.publish(grasp_pose)
        print("Calculating grasp pose for {}".format(self.obj_to_grasp.name))
        self.next_state = "wait_for_grasp"
        self.print = True
    
    def wait_for_grasp(self):
        if self.print:
            print("Wait for fetch to grasp the object")
            self.print = False

    
        


            


if __name__ == '__main__':
    rospy.init_node("driver_node")
    foo = StateMachine()
    while not rospy.is_shutdown():
        foo.run()

