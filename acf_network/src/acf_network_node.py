#!/home/cuhsailus/anaconda3/envs/acf/bin/python
import os
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
from geometry_msgs.msg import PoseStamped


RGB_IMAGE_TOPIC = "/head_camera/rgb/image_raw"
DEPTH_IMAGE_TOPIC = "/head_camera/depth/image_raw"
INPUT_MODE = "RBGD"
CHECKPOINT_PATH = "/home/cuhsailus/Desktop/ACF_fetch/src/acf_network/src/best_loss_RGBD_epoch90.pth"
DATASET = "real_world"
OUT_DIR = "/home/cuhsailus/Desktop/affordance_coordinate_frame/out/"
MASK_ALPHA = 0.6

class Object():
    def __init__(self, obj_name, kp1, ax1, kp2=None, ax2=None):
        self.id = uuid.uuid4() #unique identifier
        self.name = obj_name
        self.kp1 = kp1
        self.ax1 = ax1
        self.kp2 = kp2
        self.ax2 = ax2
    
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
        self.camera_matrix = np.array([[536.544, 0., 324.149],
                            [0., 537.666, 224.299],
                            [0., 0., 1.]])
        self.object_classes = {0: "mug", 1:"bottle", 2:"spoon/spatula", 3:"other_container"}
        self.acf_labels = {1:"container", 2:"handle", 3:"stir", 4:"scoop"}
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
        self.detect_sub = rospy.Subscriber("/ACF_Network/run_network", Bool, self.detection_callback)
        self.tf_listener = tf.TransformListener()

    def img_callback(self, rgb_msg, depth_msg):
        with self._lock:
            self._rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg)
            self._rgb_img = cv2.cvtColor(self._rgb_img, cv2.COLOR_BGR2RGB)
            self._depth_img = self.bridge.imgmsg_to_cv2(depth_msg)
            t = self.tf_listener.getLatestCommonTime('/base_link', '/head_camera_link')
            self._cam_pose = self.tf_listener.lookupTransform('/base_link', '/head_camera_link', rospy.Time(0))
    
        

    def detection_callback(self, msg):
        rospy.logwarn("Received call to run network. Running...")
        # print("IN CALLBACK")
        if self._rgb_img is None or self._depth_img is None:
            rospy.logwarn("ACF Detector hasn't gotten RGB and/or Depth imgs yet...RETURNING")
            return
        with self._lock:
            self.current_detections.clear()
            image_rgb = np.array(self._rgb_img).astype(np.float32) / 255
            image_depth = TF.to_tensor(self._depth_img.astype(np.float32) / 10).to(self.device)
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

            self.visualize(self._rgb_img.copy(), proposals, [None], final_pafs_pair)
        self.group_objects(proposals, final_pafs_pair)
        self.publish_detections()
        
    def publish_detections(self):
        self.scene_obj_msgs = []
        for obj in self.current_detections:
            msg = SceneObject()
            # trans_camera_link = np.array([obj.kp1[0], obj.kp1[1], obj.kp1[2]]).reshape((3,1))
            # rot_camera_link = Rotation.from_euler('xyz', obj.ax1, degrees=False).as_matrix()
            # homogeneous_mat = np.hstack((rot_camera_link, trans_camera_link))
            # homogeneous_mat = np.vstack((homogeneous_mat, np.array([0,0,0,1]).reshape((1,4))))

            # kp_loc_base_link = np.dot(self._cam_pose, homogeneous_mat)
            # msg.kp1.position.x = kp_loc_base_link[0,3]
            # msg.kp1.position.y = kp_loc_base_link[1,3]
            # msg.kp1.position.z = kp_loc_base_link[2,3]
            # rot = Rotation.from_matrix(homogeneous_mat[0:3, 0:3])
            # q1 = rot.as_quat()
            # msg.kp1.orientation.x = q1[0]
            # msg.kp1.orientation.y = q1[1]
            # msg.kp1.orientation.z = q1[2]
            # msg.kp1.orientation.w = q1[3]


            pose_camera_link = PoseStamped()
            pose_camera_link.header.frame_id = '/head_camera_link'
            pose_camera_link.header.stamp = rospy.Time.now()
            pose_camera_link.pose.position.x = obj.kp1[0]
            pose_camera_link.pose.position.y = obj.kp1[1]
            pose_camera_link.pose.position.z = obj.kp1[2]
            rot1 = Rotation.from_euler('xyz', obj.ax1, degrees=False)
            q1 = rot1.as_quat()
            pose_camera_link.pose.orientation.x = q1[0]
            pose_camera_link.pose.orientation.y = q1[1]
            pose_camera_link.pose.orientation.z = q1[2]
            pose_camera_link.pose.orientation.w = q1[3]
            self.tf_listener.waitForTransform('/base_link', '/head_camera_link', pose_camera_link.header.stamp, rospy.Duration(5))
            if self.tf_listener.canTransform('/base_link', '/head_camera_link', pose_camera_link.header.stamp):
                rospy.logwarn("Able to do transform!")
                pose_base_link = self.tf_listener.transformPose('/base_link', pose_camera_link)
                rospy.logwarn(pose_base_link)
            else:
                rospy.logfatal("Could not transfrom kp pose from head_camera_link to base_link")
            msg.name = obj.name
            msg.kp1 = pose_base_link
            # if obj.kp2 is not None:
            #     pose_camera_link = PoseStamped()
            #     pose_camera_link.header.frame_id = '/head_camera_link'
            #     pose_camera_link.header.stamp = rospy.Time.now()
            #     pose_camera_link.pose.position.x = obj.kp2[0]
            #     pose_camera_link.pose.position.y = obj.kp2[1]
            #     pose_camera_link.pose.position.z = obj.kp2[2]
            #     rot1 = Rotation.from_euler('xyz', obj.ax2, degrees=False)
            #     q1 = rot1.as_quat()
            #     pose_camera_link.pose.orientation.x = q1[0]
            #     pose_camera_link.pose.orientation.y = q1[1]
            #     pose_camera_link.pose.orientation.z = q1[2]
            #     pose_camera_link.pose.orientation.w = q1[3]
            #     pose_base_link = self.tf_listener.transformPose('/base_link', pose_camera_link)
            #     msg.kp2 = pose_base_link
            # else:
            msg.kp2 = pose_base_link
            self.scene_obj_msgs.append(msg)
        arr_msg = SceneObjectArray()
        arr_msg.scene_objects = self.scene_obj_msgs
        self.scene_obj_pub.publish(arr_msg)

        
    def group_objects(self, proposals, final_pafs_pair):
        print("Grouping objects")
        axes = proposals[0]['axis'].tolist()
        keypoints_3d = proposals[0]['keypoints_3d'].tolist()
        try:
            for i in range(len(final_pafs_pair)):
                pair = final_pafs_pair[i][0] #final_pafs_pair is list of lists
                kp1 = keypoints_3d[pair[0]]
                kp2 = keypoints_3d[pair[1]]
                ax1 = axes[pair[0]]
                ax2 = axes[pair[1]]
                keypoints_3d.remove(kp1)
                keypoints_3d.remove(kp2)
                axes.remove(ax1)
                axes.remove(ax2)
                self.current_detections.append(Object(self.object_classes[pair[-1]], kp1, ax1, kp2, ax2))
        except IndexError:
            pass

        for ax,kp in zip(axes,keypoints_3d):
            # Set remaining kps to "other_container" as a catch all
            self.current_detections.append(Object("other_container", kp, ax, None, None))
            
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

        


        
    def visualize(self, rgb_img, proposals, targets, final_paf_pairs=None):
        orig = np.array(rgb_img).transpose((2,0,1))
        
        imgs = []
        for i, (p,t) in enumerate(zip(proposals, targets)):
            num_prop = p['scores'].size(0)
            if num_prop < 1:
                imgs.append((orig, 0, orig))
            scores = p['scores'].chunk(num_prop)
            masks = p['masks'].chunk(num_prop)
            boxes = p['boxes'].chunk(num_prop)
            labels = p['labels'].chunk(num_prop)
            keypoints_3d = p['keypoints_3d']
            fx, fy, cx, cy = self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            if keypoints_3d is not None:
                keypoints_x = keypoints_3d[:, 0] / keypoints_3d[:, 2] * fx + cx
                keypoints_y = keypoints_3d[:, 1] / keypoints_3d[:, 2] * fy + cy
                keypoints = torch.stack((keypoints_x, keypoints_y), dim=1).cpu()

            axis = p['axis']
            if axis is not None:
                if t is not None:
                    target_axis_keypoints = t['axis_keypoints']
                    t_keypoints_x = target_axis_keypoints[..., 0] / target_axis_keypoints[..., 2] * fx + cx
                    t_keypoints_y = target_axis_keypoints[..., 1] / target_axis_keypoints[..., 2] * fy + cy
                    target_axis_kp = torch.stack((t_keypoints_x, t_keypoints_y), dim=2)
            for i in range(len(scores)):
                s, m, b, l = scores[i], masks[i], boxes[i], labels[i]
                s, m, b, l = s.squeeze(0).cpu().numpy(), m.squeeze().cpu().numpy(), b.squeeze(0).cpu().numpy(), \
                            l.cpu().numpy()[0]
                if keypoints_3d is not None:
                    k = keypoints[i].squeeze().cpu().numpy()
                
                #put bboxes on image
                x1, y1, x2, y2 = b
                start_pt = (x1, y1)
                end_pt = (x2,y2)
                cv2.rectangle(rgb_img,start_pt,end_pt,(255,255,255), thickness=2)
                text_xy = b[:2]
                text_xy[0] = max((x1 + x2) / 2 - 18, 0)
                if l == 2:
                    text_xy[0] = max((x1 + x2) / 2 - 25, 0)
                text_xy[1] = max(y1 - 18, 0)
                cv2.putText(rgb_img, self.classes_inverse[l], (text_xy[0],text_xy[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0,0,0))
                if keypoints_3d is not None:
                    rgb_img = cv2.circle(rgb_img, tuple(k), 4, color=(0, 0, 255), thickness=-1)
                if axis is not None:
                    length = 10
                    one_axis = axis[i].cpu().numpy()
                    center_keypoint = keypoints_3d[i].squeeze().cpu().numpy()
                    another_keypoint = center_keypoint + length * one_axis
                    project_2d_x = another_keypoint[0] / another_keypoint[2] * fx + cx
                    project_2d_y = another_keypoint[1] / another_keypoint[2] * fy + cy
                    end_point = (int(project_2d_x), int(project_2d_y))
                    rgb_img = cv2.arrowedLine(rgb_img, tuple((k).astype("int")), end_point, color=(0, 165, 255), thickness = 3, tipLength=0.2)
                            
                if axis is not None and t is not None:
                    for takp in target_axis_kp:
                        takp = takp.cpu().numpy()
                        start_point = (takp[0, 0], takp[0, 1])
                        end_point = (takp[1, 0], takp[1, 1])
                        rgb_img = cv2.arrowedLine(rgb_img, start_point, end_point, (0, 255, 0), 3, tipLength=0.2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img))
        # if not cv2.imwrite("/home/cuhsailus/Desktop/on_bot/{}.png".format(self.i), rgb_img):
        #     raise Exception("Could not write to loc")
        print("Done visualizing")
        print("###############################\n")




    
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
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        r.sleep()