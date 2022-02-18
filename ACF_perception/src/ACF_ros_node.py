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

import aff_cf_model
import acfutils.common_utils as utils
import config

import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ACF_perception.msg import SceneObject, SceneObjectArray


RGB_IMAGE_TOPIC = "/head_camera/rgb/image_raw"
DEPTH_IMAGE_TOPIC = "/head_camera/depth/image_raw"
INPUT_MODE = "RBGD"
CHECKPOINT_PATH = "best_loss_RGBD_epoch90.pth"
DATASET = "real_world"
OUT_DIR = "/home/cuhsailus/Desktop/affordance_coordinate_frame/out/"
MASK_ALPHA = 0.6

class Object():
    def __init__(self, obj_name, kp1, ax1, kp2=None, ax2=None):
        self.id = obj_name
        self.kp1 = kp1
        self.ax1 = ax1
        self.kp2 = kp2
        self.ax2 = ax2
    
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
        self.camera_matrix = np.array([[536.544, 0., 324.149],
                            [0., 537.666, 224.299],
                            [0., 0., 1.]])
        self.object_classes = {0: "mug", 1:"bottle", 2:"spoon/spatula", 3:"other_container"}
        self.acf_labels = {1:"container", 2:"handle", 3:"stir", 4:"scoop"}
        self.current_detections = []

        #ROS stuff
        rgb_sub = Subscriber(RGB_IMAGE_TOPIC, Image)
        depth_sub = Subscriber(DEPTH_IMAGE_TOPIC, Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 2, 0.5)
        ts.registerCallback(self.callback)
        self.bridge = CvBridge()
        self.i = 0
        self.image_pub = rospy.Publisher("/ACF_Detector/RGB_IN", Image, queue_size=100)
        self.scene_obj_pub = rospy.Publisher("/ACF_detector/scene_objects", SceneObjectArray, queue_size=10)


    def callback(self, rgb_msg, depth_msg):
        print("IN CALLBACK")
        self.current_detections.clear()
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg)
        image_rgb = np.array(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255
        image_depth = TF.to_tensor(depth_img.astype(np.float32) / 10).to(self.device)
        image_depth_tensor = TF.to_tensor(self.depth2normal(depth_img)).type(torch.float32)
        image_rgb_tensor = TF.to_tensor(image_rgb)
        img = torch.cat((image_rgb_tensor, image_depth_tensor), dim=0)
        img = img.unsqueeze(0)

        img = img.to(self.device)

        with torch.no_grad():
            detections, _ = self.model(img)

        proposals = utils.post_process_proposals(detections, image_depth, K=config.BOX_POSTPROCESS_NUMBER, camera_mat = self.camera_matrix)
        try:
            final_pafs_pair = utils.pafprocess(proposals, self.camera_matrix)
        except:
            rospy.WARN("paf ERROR")
            final_pafs_pair = None
        #self.visualize(rgb_img, proposals, [None], self.classes, self.camera_matrix, final_pafs_pair)
        self.group_objects(proposals, final_pafs_pair)
        self.scene_obj_msgs = []
        for obj in self.current_detections:
            msg = SceneObject()
            msg.name = obj.id
            msg.kp1.position.x = obj.kp1[0]
            msg.kp1.position.y = obj.kp1[1]
            msg.kp1.position.z = obj.kp1[2]
            rot1 = Rotation.from_euler('xyz', obj.ax1, degrees=False)
            q1 = rot1.as_quat()
            msg.kp1.orientation.x = q1[0]
            msg.kp1.orientation.y = q1[1]
            msg.kp1.orientation.z = q1[2]
            msg.kp1.orientation.w = q1[3]
            if obj.kp2 is not None:
                msg.kp2.position.x = obj.kp2[0]
                msg.kp2.position.y = obj.kp2[1]
                msg.kp2.position.z = obj.kp2[2]
                rot2 = Rotation.from_euler('xyz', obj.ax2, degrees=False)
                q2 = rot2.as_quat()
                msg.kp1.orientation.x = q2[0]
                msg.kp1.orientation.y = q2[1]
                msg.kp1.orientation.z = q2[2]
                msg.kp1.orientation.w = q2[3]
            else:
                obj.kp2 = None
            self.scene_obj_msgs.append(msg)
        arr_msg = SceneObjectArray()
        arr_msg.scene_objects = self.scene_obj_msgs
        self.scene_obj_pub.publish(arr_msg)

    
    def group_objects(self, proposals, final_pafs_pair):
        assert(len(proposals) == 1)
        axes = proposals[0]['axis'].tolist()
        keypoints_3d = proposals[0]['keypoints_3d'].tolist()
        # for p in proposals:
            # get all keypoints
            # keypoints_3d = p['keypoints_3d']
            # keypoints_u = keypoints_3d[:, 0] / keypoints_3d[:, 2] * self.camera_matrix[0, 0] + self.camera_matrix[0, 2]
            # keypoints_v = keypoints_3d[:, 1] / keypoints_3d[:, 2] * self.camera_matrix[1, 1] + self.camera_matrix[1, 2]
            # keypoints = torch.stack((keypoints_u, keypoints_v), dim=1).cpu().numpy().tolist()
            # for kp in keypoints:
            #     print(kp)
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

        for ax,kp in zip(axes,keypoints_3d):
            # Set remaining kps to "other_container" as a catch all
            self.current_detections.append(Object("other_container", kp, ax, None, None))
            

        


        
    def visualize(self, rgb_img, proposals, targets, classes, cam_int_mat, final_paf_pairs=None):
        orig = np.array(rgb_img).transpose((2,0,1))
        classes_inverse = {classes[key]: key for key in classes}
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
            fx, fy, cx, cy = cam_int_mat[0, 0], cam_int_mat[1, 1], cam_int_mat[0, 2], cam_int_mat[1, 2]
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
                cv2.putText(rgb_img, classes_inverse[l], (text_xy[0],text_xy[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0,0,0))
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
                        # draw.line([takp[0, 0], takp[0, 1], takp[1, 0], takp[1, 1]], fill='green', width=3)
                        start_point = (takp[0, 0], takp[0, 1])
                        end_point = (takp[1, 0], takp[1, 1])
                        rgb_img = cv2.arrowedLine(rgb_img, start_point, end_point, (0, 255, 0), 3, tipLength=0.2)
                    # rgb_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        # out = os.path.join(OUT_DIR, "{}.png".format(self.i))
        # print(out)
        if not cv2.imwrite("/home/cuhsailus/Desktop/on_bot/{}.png".format(self.i), rgb_img):
            raise Exception("Could not write to loc")
        print("Done visualizing")
        print("###############################\n\n")




    
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






def test_net(args):
    if args.workers > 1:
        mp.set_start_method('spawn')

    model = aff_cf_model.ACFNetwork(arch='resnet50', pretrained=True, num_classes=5,
                                    input_mode=args.input_mode, acf_head=args.acf_head, input_channel=6)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if args.checkpoint_path != '':
        state_dict = torch.load(args.checkpoint_path, map_location = device)
        model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    out_imgdir = args.out_dir + '/result_images/'
    out_evldir = args.out_dir + '/result_eval/'
    os.system('mkdir -p ' + out_imgdir)
    os.system('mkdir -p ' + out_evldir)
    os.system('rm '+out_imgdir+'*')

    # folder_name = '/home/cxt/Documents/research/affordance/fetch_experiment/real_obj_data/data/test_old_set/data/rgb/'
    for image_path in glob.glob(args.data_dir + '/*rgb.png'):
        image_rgb = np.array(cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)).astype(np.float32) / 255
        # image_rgb = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255
        image_depth_ori = cv2.imread(image_path.replace('rgb', 'depth'), -1)
        image_depth = TF.to_tensor(image_depth_ori.astype(np.float32) / 10).to(device)
        image_depth_tensor = TF.to_tensor(depth2normal(image_depth_ori)).type(torch.float32)
        image_rgb_tensor = TF.to_tensor(image_rgb)
        # image_depth_tensor = TF.to_tensor(image_depth).type(torch.float32)
        img = torch.cat((image_rgb_tensor, image_depth_tensor), dim=0)
        img = img.unsqueeze(0)

        img = img.to(device)

        with torch.no_grad():
            detections, _ = model(img)
        # camera_matrix = np.array([ [904.572, 0., 635.981],
        #                             [0., 905.295, 353.060],
        #                             [0., 0., 1.]])
        camera_matrix = np.array([[536.544, 0., 324.149],
                                    [0., 537.666, 224.299],
                                    [0., 0., 1.]])
        proposals = utils.post_process_proposals(detections, image_depth, K=config.BOX_POSTPROCESS_NUMBER, camera_mat = camera_matrix)
        final_pafs_pair = None
        # try:
        #     final_pafs_pair = utils.pafprocess(proposals, camera_matrix)
        # except:
        #     print('paf error')

        classes = {"__background__": 0,
                   "body": 1,
                   "handle": 2,
                   "stir": 3,
                   "head": 4}
        utils.vis_images(proposals, [None], [image_path], classes, camera_matrix, training=False,
                            output_dir=out_imgdir, final_pafs_pair=final_pafs_pair, save_image_name = image_path.split('/')[-1])

        print('testing output to: ' + out_imgdir)


if __name__ == '__main__':
    rospy.init_node('ACF_Detector')
    r = rospy.Rate(1)
    foo = ACF_Detector()
    while not rospy.is_shutdown():
        r.sleep()
    # args = utils.parse_args('test')
    # test_net(args)