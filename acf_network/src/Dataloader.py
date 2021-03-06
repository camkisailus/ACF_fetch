import glob
import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
import cv2
import config
from scipy.spatial.transform import Rotation as R



def collate_fn(batch):
    imgs, targets = tuple(zip(*batch))
    return torch.stack(imgs), targets


def make_data_loader(split, dataset_name, data_dir, batch_size=4, workers=4, shuffle=True, device = None):
    if dataset_name == "unreal_parts":
        dataset = ACFpaf(data_dir, split , device = device)
    elif dataset_name == 'real_world':
        dataset = ACFpaf_Real(data_dir, split, device = device)
    else:
        raise Exception("Unrecognized dataset {}".format(dataset_name))

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                         pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return loader, dataset.CLASS_LABELS


def get_labels():
    labels = {"__background__": 0,
              "body": 1,
              "handle": 2,
              "stir": 3,
              "head": 4}

    return labels

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    return np.array([x1, y1, x2, y2])

part_affinity_dict = [
    ['body', 'handle'],
    ['handle', 'body'],
    ['stir', 'head'],
    ['head', 'stir']
]

class ACFpaf(Dataset):
    def __init__(self, root, setd, use_binmasks=False, input_y=480, input_x=640, stride=8, device=None,
                 background=True):
        self.root = root
        self.setd = setd
        self.use_binmasks = use_binmasks
        self.input_y = input_y
        self.input_x = input_x
        self.stride = stride
        self.intriscM = np.array([[input_x/2, 0.0, input_x/2], [0.0, input_x/2, input_y/2], [0.0, 0.0, 1.0]])

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.images = []

        self.read_lists()
        self.CLASS_LABELS = get_labels()
        self.endpoints_offset = config.Unreal_keypoint_offset
        self.center_keypoints_offset = config.Unreal_center_keypoint_offset
        self.background = background
        if not background and '__background__' in self.CLASS_LABELS:
            self.CLASS_LABELS = {k: v - 1 for k, v in self.CLASS_LABELS.items() if k != '__background__'}

    def __getitem__(self, index):
        fx, fy, cx, cy = self.intriscM[0, 0], self.intriscM[1, 1], self.intriscM[0, 2], self.intriscM[1, 2]
        if self.setd == 'train':
            folder = 'train'
        else:
            folder = 'test'
        image_path = self.root + folder + '/' + self.images[index]
        # image_rgb = np.array(Image.open(image_path).resize(self.input_x, self.input_y).convert("RGB")).astype(np.float32) / 255
        image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255
        image_rgb = cv2.resize(image_rgb, (self.input_x, self.input_y))
        # mask = np.array(Image.open(image_path.replace('.png', '.is.png')).resize(self.input_x, self.input_y).convert("RGB")).astype(np.longlong)
        mask = cv2.cvtColor(cv2.imread(image_path.replace('.png', '.is.png')), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.input_x, self.input_y), interpolation=cv2.INTER_AREA).astype(np.longlong)
        mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
        # mask = mask[:, :, 0] << 16 + mask[:, :, 1] << 8 + mask[:, :, 2] # TODO: debug why this format not work
        image_depth = cv2.imread(image_path.replace('.png', '.depth.mm.16.png'), -1).astype(np.float32)
        image_depth = cv2.resize(image_depth,(self.input_x, self.input_y))
        image_depth_cm = image_depth / 10.0

        json_path = image_path.replace('.png', '.json')
        assert os.path.exists(json_path), "Path does not exist: {}".format(json_path)
        jsondata = self.load_jsonfile(json_path)

        bboxes, center_points, labels, instance_ids, keypoints, depth_avgs, depth_offsets = [], [], [], [], [], [], []
        axis_keypoints = []
        poses = []
        names = []
        for i, name in enumerate(jsondata['names']):
            if 'edge' in name and 'scoop4_head' in name:
                continue
            instance_id = jsondata['instance_ids'][i]
            # don't learn mug edge anymore
            if 'body' in name:
                edge_name = name.replace('body', 'edge')
                if edge_name in jsondata['names']:
                    edge_index = jsondata['names'].index(edge_name)
                    edge_id = jsondata['instance_ids'][edge_index]
                    mask[mask == edge_id] = instance_id

            if np.sum(mask == instance_id) < 25:
                continue
            class_name = name.split('_')[1]
            vaild_part = False
            for c in self.CLASS_LABELS.keys():
                if c in class_name:
                    vaild_part = True
                    name_c = name.split('_')[0] + '_' + c
                    endpoint_offset = self.endpoints_offset[name_c]
                    center_offset = self.center_keypoints_offset[name_c]
                    label = self.CLASS_LABELS[c]

            if not vaild_part:
                continue
            names.append(name)
            bbox = extract_bboxes(mask==instance_id)
            bboxes.append(bbox)
            rotm = R.from_quat(jsondata['quaternions'][i]).as_matrix()

            instance_ids.append(instance_id)

            labels.append(label)
            scale = jsondata['scales'][i]
            keypoint1 = np.dot(rotm, np.array(endpoint_offset[:3]) * scale) + \
                        jsondata['keypoints_3d'][i][8]
            keypoint2 = np.dot(rotm, np.array(endpoint_offset[3:6]) * scale) + \
                        jsondata['keypoints_3d'][i][8]

            axis_keypoints.append([keypoint1, keypoint2])

            center_3d_keypoints = np.dot(rotm, np.array(center_offset[:3]) * scale) + \
                                  jsondata['keypoints_3d'][i][8]
            center_points.append(center_3d_keypoints)
            pose = list(center_3d_keypoints) + list(jsondata['quaternions'][i])
            poses.append(pose)

        target_pafs = []
        for i, name in enumerate(names):
            class_name = name.split('_')[1]
            target_paf = np.array([0, 0])
            for pair in part_affinity_dict:
                if pair[0] in class_name and name.replace(pair[0], pair[1]) in names:
                    target_name = name.replace(pair[0], pair[1])
                    target_index = names.index(target_name)
                    target_center_3d_keypoints = center_points[target_index]
                    keypoints_x = target_center_3d_keypoints[0] / target_center_3d_keypoints[2] * fx + cx
                    keypoints_y = target_center_3d_keypoints[1] / target_center_3d_keypoints[2] * fy + cy
                    target_paf_keypoint = np.array([keypoints_x, keypoints_y])
                    center_3d_keypoints = center_points[i]
                    keypoints_x = center_3d_keypoints[0] / center_3d_keypoints[2] * fx + cx
                    keypoints_y = center_3d_keypoints[1] / center_3d_keypoints[2] * fy + cy
                    keypoint = np.array([keypoints_x, keypoints_y])
                    target_paf = target_paf_keypoint - keypoint
                    target_paf = target_paf / np.linalg.norm(target_paf)
                    break
                use_axis_as_paf = False
                if 'body' in name and 'mug' not in name:
                    use_axis_as_paf = True
                if 'hammer' in name:
                    use_axis_as_paf = True
                if use_axis_as_paf:
                    keypoint1 = axis_keypoints[i][0]
                    keypoint2 = axis_keypoints[i][1]
                    keypoints_x = keypoint1[0] / keypoint1[2] * fx + cx
                    keypoints_y = keypoint1[1] / keypoint1[2] * fy + cy
                    keypoint = np.array([keypoints_x, keypoints_y])
                    keypoints_x = keypoint2[0] / keypoint2[2] * fx + cx
                    keypoints_y = keypoint2[1] / keypoint2[2] * fy + cy
                    target_paf_keypoint = np.array([keypoints_x, keypoints_y])
                    target_paf = target_paf_keypoint - keypoint
                    target_paf = target_paf / np.linalg.norm(target_paf)
                    break
            target_pafs.append(target_paf)

        image_rgb = TF.to_tensor(image_rgb)
        image_depth_normal = TF.to_tensor(ACFpaf_Real.depth2normal(image_depth)).type(torch.float32)
        # image_depth = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min())
        # image_depth = TF.to_tensor(image_depth).type(torch.float32)
        # image_depth = image_depth.repeat(3, 1, 1)
        mask = torch.tensor(mask, dtype=torch.long)
        labels = torch.tensor(np.array(labels, dtype=np.int), dtype=torch.long)
        instance_ids = torch.tensor(np.array(instance_ids, dtype=np.int), dtype=torch.long)
        bboxes = torch.tensor(np.array(bboxes, dtype=np.int), dtype=torch.float)
        poses = torch.tensor(np.array(poses, dtype=np.float), dtype=torch.float)
        image_depth_cm_tensor = TF.to_tensor(image_depth_cm).type(torch.float32)
        axis_keypoints = torch.tensor(np.array(axis_keypoints, dtype=np.float), dtype=torch.float)
        target_pafs = torch.tensor(np.array(target_pafs, dtype=np.float), dtype=torch.float)
        img = torch.cat((image_rgb, image_depth_normal), dim=0)

        targets = {'boxes': bboxes,
                   'labels': labels,
                   'instance_ids': instance_ids,
                   'masks': mask,  # bin_masks
                   'frame': poses,
                   'axis_keypoints': axis_keypoints,
                   'target_pafs': target_pafs,
                   'img_file': image_path,
                   'names':names,
                   'ori_image_depth': image_depth_cm_tensor,
                   'CameraMatrix': self.intriscM}
        return img, targets

    def read_lists(self):
        image_path = os.path.join(self.root, "rgb_files_{}.txt".format(self.setd))

        assert os.path.exists(image_path), "Path does not exist: {}".format(image_path)

        self.images = [line.strip() for line in open(image_path, 'r')]

    def load_jsonfile(self, json_path):
        """
                Loads the data from a json file.
                If there are no objects of interest, then load all the objects.
                """
        with open(json_path) as data_file:
            data = json.load(data_file)

        points_keypoints_2d = []
        points_keypoints_3d = []
        pointsBoxes = []
        boxes = []
        names = []
        scales = []
        quaternions = []
        instance_ids = []

        for i_line in range(len(data['objects'])):
            info = data['objects'][i_line]
            name = info['class']
            if 'scoop' in name or 'hammer' in name:
                name = name.replace('handle', 'stir')
            instance_name = name.split('_')[0]
            class_name = name.split('_')[1]
            name = instance_name + '_' + class_name
            names.append(name)
            box = info['bounding_box']
            boxToAdd = []

            boxToAdd.append(float(box['top_left'][0]))
            boxToAdd.append(float(box['top_left'][1]))
            boxToAdd.append(float(box["bottom_right"][0]))
            boxToAdd.append(float(box['bottom_right'][1]))
            boxes.append(boxToAdd)

            boxpoint = [[boxToAdd[1], boxToAdd[0]], [boxToAdd[3], boxToAdd[0]],
                        [boxToAdd[1], boxToAdd[2]], [boxToAdd[3], boxToAdd[2]]]  # use x,y to index the bbox

            pointsBoxes.append(boxpoint)

            scale = np.linalg.norm(np.array(info['pose_transform'])[:3, :3], axis=0).mean()
            scales.append(scale)

            # 2d projected key points
            point2dToAdd = []
            pointdata = info['projected_cuboid']
            for p in pointdata:
                point2dToAdd.append([p[0], p[1]])  # change x,y index to row,col index

            # Get the centroids
            pcenter = info['projected_cuboid_centroid']

            point2dToAdd.append([pcenter[0], pcenter[1]])
            points_keypoints_2d.append(point2dToAdd)

            # 2d projected key points
            point3dToAdd = []
            pointdata = info['cuboid']
            for p in pointdata:
                point3dToAdd.append([p[0], p[1], p[2]])  # change x,y index to row,col index

            # Get the centroids
            pcenter = info['cuboid_centroid']

            point3dToAdd.append([pcenter[0], pcenter[1], pcenter[2]])
            points_keypoints_3d.append(point3dToAdd)

            # Get roatation matrix
            quaternion_xyzw = info['quaternion_xyzw']
            quaternions.append(quaternion_xyzw)

            # Get instance_id
            instance_id = info['instance_id']
            instance_ids.append(instance_id)

        return {
            "scales": scales,
            "names": names,
            "bbox": pointsBoxes,
            "keypoints_2d": points_keypoints_2d,  # 8 keypoints + center
            "keypoints_3d": points_keypoints_3d,  # 8 keypoints + center
            "quaternions": quaternions,
            "instance_ids": instance_ids
        }

    def __len__(self):
        return len(self.images)

class ACFpaf_Real(Dataset):
    def __init__(self, root, setd, input_y=480, input_x=640, stride=8, device=None,
                 background=True):
        self.root = root
        self.setd = setd
        self.input_y = input_y
        self.input_x = input_x
        self.stride = stride
        self.intriscM = np.array([[536.544, 0., 324.149], [0., 537.666, 224.299], [0., 0., 1.]])

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.images = []

        self.read_lists()
        self.CLASS_LABELS = get_labels()
        self.background = background
        if not background and '__background__' in self.CLASS_LABELS:
            self.CLASS_LABELS = {k: v - 1 for k, v in self.CLASS_LABELS.items() if k != '__background__'}

    def read_lists(self):
        image_list_path = os.path.join(self.root, "real_rgb_files_{}.txt".format(self.setd))
        if not os.path.exists(image_list_path):
            rgb_list = glob.glob(os.path.join(self.root, self.setd) + '/**/rgb/*.png')
            depth_list = glob.glob(os.path.join(self.root, self.setd) + '/**/depth/*.png')

            assert len(rgb_list) == len(depth_list), 'RGB and Depth should have same number'
            self.images = rgb_list
            image_list = open(image_list_path, 'w')
            for rgb in rgb_list:
                image_list.write(rgb + '\n')
            image_list.close()
        else:
            self.images = [line.strip() for line in open(image_list_path, 'r')]

    def __getitem__(self, index):
        fx, fy, cx, cy = self.intriscM[0, 0], self.intriscM[1, 1], self.intriscM[0, 2], self.intriscM[1, 2]
        image_path = self.images[index]
        if 'rgb' not in image_path.rsplit('/')[-1]:
            mask_filename = image_path.replace('rgb', 'mask').replace('.png', '_mask.png')
        else:
            mask_filename = image_path.replace('rgb', 'mask')
        mask = cv2.imread(mask_filename, -1)
        mask_id_path = os.path.join(os.path.dirname(image_path).replace('rgb', 'mask'),'mask_annotation.json')
        with open(mask_id_path) as f:
            instance_dict = json.load(f)

        image_depth = cv2.imread(image_path.replace('rgb', 'depth'), -1)

        json_path = image_path.replace('_rgb', '').replace('.png', '.json').replace('rgb', 'acf_labels')
        assert os.path.exists(json_path), "Path does not exist: {}".format(json_path)
        with open(json_path) as data_file:
            jsondata = json.load(data_file)

        names, labels, instance_ids = [], [], []
        bboxes, axis_keypoints, center_points, poses = [], [], [], []
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = np.zeros_like(mask)
        for obj_name in jsondata:
            obj = jsondata[obj_name]
            for part in obj['parts']:
                name = obj_name+'_'+part
                instance_id = instance_dict[name]
                instance_mask = np.array(mask == instance_id).astype(np.uint8)
                instance_mask = cv2.morphologyEx(instance_mask, cv2.MORPH_OPEN, kernel)
                instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
                if np.sum(instance_mask) < 50:
                    continue
                clean_mask[instance_mask==1] = instance_id

                label = self.CLASS_LABELS[part]
                names.append(name)
                instance_ids.append(instance_id)
                labels.append(label)
                bbox = extract_bboxes(instance_mask)
                bboxes.append(bbox)
                keypoint0 = obj['parts'][part][0]
                keypoint1 = obj['parts'][part][1]
                center_point = obj['parts'][part][2]
                axis_keypoints.append([keypoint0, keypoint1])
                center_points.append(center_point)
                poses.append(center_point+[0, 0, 0, 0])

        target_pafs = []
        for i, name in enumerate(names):
            class_name = name.split('_')[-1]
            target_paf = np.array([0, 0])
            for pair in part_affinity_dict:
                if pair[0] in class_name and name.replace(pair[0], pair[1]) in names:
                    target_name = name.replace(pair[0], pair[1])
                    target_index = names.index(target_name)
                    target_center_3d_keypoints = center_points[target_index]
                    keypoints_x = target_center_3d_keypoints[0] / target_center_3d_keypoints[2] * fx + cx
                    keypoints_y = target_center_3d_keypoints[1] / target_center_3d_keypoints[2] * fy + cy
                    target_paf_keypoint = np.array([keypoints_x, keypoints_y])
                    center_3d_keypoints = center_points[i]
                    keypoints_x = center_3d_keypoints[0] / center_3d_keypoints[2] * fx + cx
                    keypoints_y = center_3d_keypoints[1] / center_3d_keypoints[2] * fy + cy
                    keypoint = np.array([keypoints_x, keypoints_y])
                    target_paf = target_paf_keypoint - keypoint
                    target_paf = target_paf / np.linalg.norm(target_paf)
                    break
                use_axis_as_paf = False
                if 'bottle' in name or 'bowl' in name:
                    use_axis_as_paf = True

                if use_axis_as_paf:
                    keypoint1 = axis_keypoints[i][0]
                    keypoint2 = axis_keypoints[i][1]
                    keypoints_x = keypoint1[0] / keypoint1[2] * fx + cx
                    keypoints_y = keypoint1[1] / keypoint1[2] * fy + cy
                    keypoint = np.array([keypoints_x, keypoints_y])
                    keypoints_x = keypoint2[0] / keypoint2[2] * fx + cx
                    keypoints_y = keypoint2[1] / keypoint2[2] * fy + cy
                    target_paf_keypoint = np.array([keypoints_x, keypoints_y])
                    target_paf = target_paf_keypoint - keypoint
                    target_paf = target_paf / np.linalg.norm(target_paf)
                    break
            target_pafs.append(target_paf)
        # image = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)
        # for b in bboxes:
        #     start_point = (b[0], b[1])
        #     end_point = (b[2], b[3])
        #     color = (255, 0, 0)
        #     thickness = 2
        #     image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # cv2.imwrite('test.png',image)
        image_rgb = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255
        # image_depth_normalized = (image_depth - image_depth.min()) / (image_depth.max() - image_depth.min())
        image_rgb = TF.to_tensor(image_rgb)
        image_depth_normal = TF.to_tensor(self.depth2normal(image_depth)).type(torch.float32)
        image_depth = TF.to_tensor(image_depth.astype(np.float32)/10)
        # image_depth = image_depth.repeat(3, 1, 1)
        mask = torch.tensor(clean_mask, dtype=torch.long)
        labels = torch.tensor(np.array(labels, dtype=np.int), dtype=torch.long)
        instance_ids = torch.tensor(np.array(instance_ids, dtype=np.int), dtype=torch.long)
        bboxes = torch.tensor(np.array(bboxes, dtype=np.int), dtype=torch.float)
        poses = torch.tensor(np.array(poses, dtype=np.float), dtype=torch.float)*100
        axis_keypoints = torch.tensor(np.array(axis_keypoints, dtype=np.float), dtype=torch.float)*100
        target_pafs = torch.tensor(np.array(target_pafs, dtype=np.float), dtype=torch.float)
        img = torch.cat((image_rgb, image_depth_normal), dim=0)

        targets = {'boxes': bboxes,
                   'labels': labels,
                   'instance_ids': instance_ids,
                   'masks': mask,  # bin_masks
                   'frame': poses,
                   'axis_keypoints': axis_keypoints,
                   'target_pafs': target_pafs,
                   'img_file': image_path,
                   'names': names,
                   'ori_image_depth': image_depth,
                   'CameraMatrix': self.intriscM}
        return img, targets

    @staticmethod
    def depth2normal(d_im):
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

    def __len__(self):
        return len(self.images)