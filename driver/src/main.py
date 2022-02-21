import rospy
from std_msgs.msg import Bool
import time


class ACF_Driver():
    def __init__(self):
        self.running = True
        self.run_net_pub = rospy.Publisher("/ACF_Network/run_network", Bool, queue_size=1)
    
    def main(self):
        ans = input("Run network? (y/n): ")
        if ans == 'y':
            self.run_net_pub.publish(True)
        elif ans == 'n':
            self.running = False
            
            


if __name__ == '__main__':
    rospy.init_node("driver_node")
    foo = ACF_Driver()
    while foo.running and not rospy.is_shutdown():
        foo.main()

