#!/home/cuhsailus/anaconda3/envs/acf/bin/python
import os
from matplotlib.colors import PowerNorm
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.multiprocessing as mp
import scipy.io as scio
import cv2
import glob
from PIL import Image
from scipy.spatial.transform import Rotation
import threading
import uuid

import aff_cf_model
import acfutils.common_utils as utils
import config

import tf
import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from acf_network.msg import SceneObject, SceneObjectArray
from geometry_msgs.msg import PoseStamped, PointStamped


RGB_IMAGE_TOPIC = "/head_camera/rgb/image_raw"
DEPTH_IMAGE_TOPIC = "/head_camera/depth_registered/image_raw"
INPUT_MODE = "RBGD"
CHECKPOINT_PATH = "/home/cuhsailus/Desktop/ACF_fetch/src/acf_network/src/best_loss_RGBD_13.pth"
DATASET = "real_world"
OUT_DIR = "/home/cuhsailus/Desktop/affordance_coordinate_frame/out/"
MASK_ALPHA = 0.6

class Object():
    def __init__(self, obj_name, kp1, ax1, bbox, kp2=None, ax2=None):
        self.id = uuid.uuid4() #unique identifier
        self.name = obj_name
        self.kp1 = kp1
        self.ax1 = ax1
        self.kp2 = kp2
        self.ax2 = ax2
        self.bbox = bbox
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        k1_eq = self.kp1 == other.kp1
        k2_eq = self.kp2 == other.kp2
        return k1_eq and k2_eq

class ACF_Detector():
    def __init__(self):
        #model stuff
        self.model = aff_cf_model.ACFNetwork(arch='resnet50', pretrained=True, num_classes=5,
                                    input_mode="RGBD", input_channel=6)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dict = torch.load(CHECKPOINT_PATH, map_location = self.device)
        self.model.load_state_dict(self.state_dict['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.classes = {"__background__": 0,
                   "body": 1,
                   "handle": 2,
                   "stir": 3,
                   "head": 4}
        self.classes_inverse = {self.classes[key]: key for key in self.classes}
        self.camera_matrix = np.array( [[527.1341414037195, 0.0, 323.8974379222906,],
                                [0.0, 525.9099904918304, 227.2282369544078],
                                [0.0, 0.0, 1.0]])
        
        # np.array([[536.544, 0., 324.149],
        #                     [0., 537.666, 224.299],
        #                     [0., 0., 1.]])
        self.object_classes = {0: "mug", 1:"bottle", 2:"spoon/spatula", 3:"other_container"}
        self.current_detections = []
        self._rgb_img = None
        self._depth_img = None
        self._cam_pose = None
        self._lock = threading.RLock()

        #ROS stuff
        rgb_sub = Subscriber(RGB_IMAGE_TOPIC, Image)
        depth_sub = Subscriber(DEPTH_IMAGE_TOPIC, Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 2, 0.5)
        ts.registerCallback(self.img_callback)
        self.bridge = CvBridge()
        self.i = 0
        self.image_pub = rospy.Publisher("/ACF_Network/detections/Image", Image, queue_size=10)
        self.scene_obj_pub = rospy.Publisher("/ACF_Network/detections/scene_objects", SceneObjectArray, queue_size=10)
        self.ready_pub = rospy.Publisher("/ACF_Network/ready_for_detections", Bool, queue_size=10)
        self.notified_user = False
        self.detect_sub = rospy.Subscriber("/ACF_Network/run_network", Bool, self.detection_callback)
        self.tf_listener = tf.TransformListener()

    def img_callback(self, rgb_msg, depth_msg):
        if not self.notified_user:
            rospy.logwarn("Got images")
            self.notified_user = True
        with self._lock:
            self._rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg)
            # self._rgb_img = cv2.cvtColor(self._rgb_img, cv2.COLOR_BGR2RGB)
            
            dtype_class, channels = np.float32, 1
            dtype = np.dtype(dtype_class)
            dtype = dtype.newbyteorder('>' if depth_msg.is_bigendian else '<')
            shape = (depth_msg.height, depth_msg.width, channels)
            data = np.fromstring(depth_msg.data, dtype=dtype).reshape(shape) * 1000 # in millimeters
            data.strides = (
                depth_msg.step,
                dtype.itemsize * channels,
                dtype.itemsize
            )

            self._depth_img = data.astype('uint16')

            # self._depth_img = self.bridge.imgmsg_to_cv2(depth_msg)
           
            # t = self.tf_listener.getLatestCommonTime('/base_link', '/head_camera_rgb_optical_frame')
            self._cam_pose = self.tf_listener.lookupTransform('/base_link', '/head_camera_rgb_optical_frame', rospy.Time(0))
            cv2.imwrite("/home/cuhsailus/Desktop/on_bot/rgb_{}.png".format(self.i), self._rgb_img)
            cv2.imwrite("/home/cuhsailus/Desktop/on_bot/depth_{}.png".format(self.i), self._depth_img)
            self.i+=1
        # self.run_detections()
            # if not self.notified_user:
            #     self.ready_pub.publish(True)
            #     rospy.logwarn("Notified user")
            #     self.notified_user = True
        
    def run_detections(self):
        with self._lock:
            self.current_detections.clear()
            image_rgb = np.array(self._rgb_img).astype(np.float32) / 255
            image_depth = TF.to_tensor(self._depth_img.astype(np.float32)/ 10).to(self.device)
            image_depth_tensor = TF.to_tensor(self.depth2normal(self._depth_img)).type(torch.float32)
            image_rgb_tensor = TF.to_tensor(image_rgb)
            img = torch.cat((image_rgb_tensor, image_depth_tensor), dim=0)
            img = img.unsqueeze(0)

            img = img.to(self.device)

            with torch.no_grad():
                detections, _ = self.model(img)


            proposals = utils.post_process_proposals(detections, image_depth, img_shape=(480,640), K=config.BOX_POSTPROCESS_NUMBER, camera_mat = self.camera_matrix)

            try:
                final_pafs_pair = utils.pafprocess(proposals, self.camera_matrix)
            except:
                rospy.logwarn("paf ERROR")
                final_pafs_pair = None
            self.group_objects(proposals, final_pafs_pair)

            self.visualize(self._rgb_img.copy())
        
        self.publish_detections()


    def detection_callback(self, msg):
        rospy.logwarn("Received call to run network. Running...")
        # print("IN CALLBACK")
        if self._rgb_img is None or self._depth_img is None:
            rospy.logwarn("ACF Detector hasn't gotten RGB and/or Depth imgs yet...RETURNING")
            return
        self.run_detections()
        
        
    def publish_detections(self):
        self.scene_obj_msgs = []

        # Invalid pose msg since ros doesn't allow optional message fields
        # Used for objects with only 1 keypoint (bowl & other_container)
        none_pose_msg = PoseStamped() 
        none_pose_msg.header.frame_id = 'None'
        none_pose_msg.header.stamp = rospy.Time.now()
        none_pose_msg.pose.position.x = 0
        none_pose_msg.pose.position.y = 0
        none_pose_msg.pose.position.z = 0
        none_pose_msg.pose.orientation.x = 0 
        none_pose_msg.pose.orientation.y = 0
        none_pose_msg.pose.orientation.z = 0
        none_pose_msg.pose.orientation.w = 1

        for obj in self.current_detections:
            kp1_pose_camera_link = PoseStamped()
            kp1_pose_camera_link.header.frame_id = '/head_camera_rgb_optical_frame'
            kp1_pose_camera_link.header.stamp = rospy.Time.now()
            kp1_pose_camera_link.pose.position.x = obj.kp1[0] / 100.0 # m -> cm conversion
            kp1_pose_camera_link.pose.position.y = obj.kp1[1] / 100.0
            kp1_pose_camera_link.pose.position.z = obj.kp1[2] / 100.0
            rot1 = Rotation.from_euler('xyz', obj.ax1, degrees=False)
            q1 = rot1.as_quat()
            kp1_pose_camera_link.pose.orientation.x = q1[0]
            kp1_pose_camera_link.pose.orientation.y = q1[1]
            kp1_pose_camera_link.pose.orientation.z = q1[2]
            kp1_pose_camera_link.pose.orientation.w = q1[3]
            if obj.kp2 is not None:
                kp2_pose_camera_link = PoseStamped()
                kp2_pose_camera_link.header.frame_id = '/head_camera_rgb_optical_frame'
                kp2_pose_camera_link.pose.position.x = obj.kp2[0] / 100
                kp2_pose_camera_link.pose.position.y = obj.kp2[1] / 100 
                kp2_pose_camera_link.pose.position.z = obj.kp2[2] / 100 
                rot1 = Rotation.from_euler('xyz', obj.ax2, degrees=False)
                q1 = rot1.as_quat()
                kp2_pose_camera_link.pose.orientation.x = q1[0]
                kp2_pose_camera_link.pose.orientation.y = q1[1]
                kp2_pose_camera_link.pose.orientation.z = q1[2]
                kp2_pose_camera_link.pose.orientation.w = q1[3]
            else:
                kp2_pose_camera_link = None
            
            self.tf_listener.waitForTransform('/base_link', '/head_camera_rgb_optical_frame', kp1_pose_camera_link.header.stamp, rospy.Duration(5))
            if self.tf_listener.canTransform('/base_link', '/head_camera_rgb_optical_frame', kp1_pose_camera_link.header.stamp):
                kp1_pose_base_link = self.tf_listener.transformPose('/base_link', kp1_pose_camera_link)
                if kp2_pose_camera_link is not None:
                    kp2_pose_base_link = self.tf_listener.transformPose('/base_link', kp2_pose_camera_link) 
            else:
                rospy.logfatal("Could not transfrom kp pose from head_camera_link to base_link")

            msg = SceneObject()
            msg.name = obj.name
            msg.kp1 = kp1_pose_base_link
            if kp2_pose_camera_link is not None:
                msg.kp2 = kp2_pose_base_link 
            else:
                msg.kp2 = none_pose_msg
            self.scene_obj_msgs.append(msg)
        arr_msg = SceneObjectArray()
        arr_msg.scene_objects = self.scene_obj_msgs
        self.scene_obj_pub.publish(arr_msg)

        
    def group_objects(self, proposals, final_pafs_pair):
        # print("Grouping objects")
        axes = proposals[0]['axis'].tolist()
        keypoints_3d = proposals[0]['keypoints_3d'].tolist()
        bboxes = proposals[0]['boxes'].tolist()
        # try:
        #     for i in range(len(final_pafs_pair)):
        #         pair = final_pafs_pair[i][0] #final_pafs_pair is list of lists
        #         kp1 = keypoints_3d[pair[0]]
        #         kp2 = keypoints_3d[pair[1]]
        #         ax1 = axes[pair[0]]
        #         ax2 = axes[pair[1]]
        #         bbox1 = bboxes[pair[0]]
        #         bbox2 = bboxes[pair[1]]
        #         bbox = (min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3]))
        #         keypoints_3d.remove(kp1)
        #         keypoints_3d.remove(kp2)
        #         axes.remove(ax1)
        #         axes.remove(ax2)
        #         bboxes.remove(bbox1)
        #         bboxes.remove(bbox2)
        #         self.current_detections.append(Object(self.object_classes[pair[-1]]+str(len(self.current_detections)), kp1, ax1, bbox, kp2, ax2))
        # except IndexError:
        #     pass
        i = 0
        for ax, kp in zip(axes,keypoints_3d):
            # Set remaining kps to "other_container" as a catch all
            bbox = bboxes[i]
            self.current_detections.append(Object("OC_"+str(i), kp, ax, bbox, None, None))
            i+=1
            
    def proposal_to_grasp(self, proposals, final_pafs_pair, cam_pose):
        # TODO: Get camera pose from TF
        grasp_pose = {obj: [] for obj in self.current_detections}
        error_dict = {obj: [] for obj in self.current_detections}
        for (p,f) in zip(proposals, final_pafs_pair):
            keypoints = p['keypoints_3d'].numpy()
            keypoints = cam_pose[:3,:3].dot(keypoints.transpose()) + cam_pose[:3,3,None]*1e2
            keypoints = keypoints.transpose()
            axis = p['axis'].numpy()
            axis = cam_pose[:3,:3].dot(axis.transpose()).transpose()
            labels = p['labels'].numpy()
            distance_errors = p['distance_errors']
            angle_errors = p['angle_errors']
            for i in range(len(self.current_detections)):
                k = list(keypoints[i])
                a = list(axis[i])
                label = [labels[i]]
                grasp_pose[self.current_detections[i]].append(k+a+label+[i])
                error_dict[self.current_detections[i]].append([distance_errors[i]] + [angle_errors[i]] + label)
            for obj in grasp_pose:
                if len(grasp_pose[obj]) != 0:
                    pose = np.array(grasp_pose[obj])
                else:
                    pose = []
                if obj.name == 'mug' or obj.name == 'spoon/spatula':
                    if len(grasp_pose[obj]) >= 2:
                        find = False
                        p1 = set(pose[:, -1])
                        for pair in f:
                            p2 = set(pair[:2])
                            if p1 == p2:
                                find = True
                                break
                        if not find:
                            pose = []
                    else:
                        pose = []
                if pose == []:
                    grasp_pose[obj] = pose
                else:
                    grasp_pose[obj] = pose[:, :-1]
        return grasp_pose, error_dict

        


        
    def visualize(self, rgb_img):
        axis_length = 10
        fx, fy, cx, cy = self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            

        for i, obj in enumerate(self.current_detections):
            if i > 5:
                break
            kp1 = obj.kp1
            kpx = kp1[0] / kp1[2] * fx + cx
            kpy = kp1[1] / kp1[2] * fy + cy        
            rgb_img = cv2.circle(rgb_img, (int(kpx), int(kpy)), 4, color=(0, 0, 255), thickness=-1)
            ax1 = obj.ax1
            another_kp = [kp1[i] + axis_length*ax1[i] for i in range(3)]
            project_2d_x = another_kp[0] / another_kp[2] * fx + cx
            project_2d_y = another_kp[1] / another_kp[2] * fy + cy
            end_point = (int(project_2d_x), int(project_2d_y))
            rgb_img = cv2.arrowedLine(rgb_img, (int(kpx), int(kpy)), end_point, color=(0, 165, 255), thickness = 3, tipLength=0.2)

            if obj.kp2 is not None:
                kp2 = obj.kp2
                kpx = kp2[0] / kp2[2] * fx + cx
                kpy = kp2[1] / kp2[2] * fy + cy
                rgb_img = cv2.circle(rgb_img, (int(kpx), int(kpy)), 4, color=(0, 0, 255), thickness=-1)
                ax2 = obj.ax2
                another_keypoint = [kp2[i] + axis_length * ax2[i] for i in range(3)]
                project_2d_x = another_keypoint[0] / another_keypoint[2] * fx + cx
                project_2d_y = another_keypoint[1] / another_keypoint[2] * fy + cy
                end_point = (int(project_2d_x), int(project_2d_y))
                rgb_img = cv2.arrowedLine(rgb_img, (int(kpx), int(kpy)), end_point, color=(0, 165, 255), thickness = 3, tipLength=0.2)
        
            bbox = obj.bbox
            x1, y1, x2, y2 = bbox
            start_pt = tuple((int(x1), int(y1)))
            end_pt = tuple((int(x2), int(y2)))
            rgb_img = cv2.rectangle(rgb_img, start_pt, end_pt , (255,255,255), thickness=2)
            text_xy = [0, 0]
            text_xy[0] = int(max((x1 + x2) / 2 - 18, 0))
            text_xy[1] = int(max(y1 - 18, 0))
            rgb_img = cv2.putText(rgb_img, obj.name, (text_xy[0],text_xy[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0,255,0)) 
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img))
        # if not cv2.imwrite("/home/cuhsailus/Desktop/on_bot/{}.png".format(self.i), rgb_img):
        #     raise Exception("Could not write to loc")
        rospy.logwarn("Done visualizing")

    def depth2normal(self, d_im):
        d_im = d_im.astype("float32")
        # zy, zx = np.gradient(d_im)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
        zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # offset and rescale values to be in 0-1
        normal += 1
        normal /= 2
        return normal


if __name__ == '__main__':
    rospy.init_node('ACF_Network')
    foo = ACF_Detector()
    r = rospy.Rate(5)
    while not rospy.is_shutdown():
        r.sleep()
