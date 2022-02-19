import rospy
from std_msgs.msg import Bool


class ACF_Driver():
    def __init__(self):
        self.run_net_pub = rospy.Publisher("/ACF_Network/run_network", Bool, queue_size=1)
        self.main()
    
    def main(self):
        input("Press enter to run the ACF network")
        self.run_net_pub.publish(True)


if __name__ == '__main__':
    rospy.init_node("driver_node")
    foo = ACF_Driver()
    while not rospy.is_shutdown():
        pass

