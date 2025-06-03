import os
import torch
import torchvision
import cv2
import json
import copy
import numpy as np
from pycocotools.coco import COCO
from preprocessing import load_img, process_bbox, augmentation, get_bbox


class DEX_YCB_SF(torch.utils.data.Dataset):

    def __init__(self, base_path, data_split):
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.data_split = data_split if data_split == "train" else "test"
        self.split = self.data_split
        self.base_path = base_path
        self.annot_path = os.path.join(self.base_path, "annotations")
        self.root_joint_idx = 9

        self.input_img_shape = [224, 224]
        self.name = "DEX_YCB_SF"
        
        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(os.path.join(self.annot_path, "DEX_YCB_s0_{}_data.json".format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            img_path = os.path.join(self.base_path, img["file_name"])
            
            img_shape = (img["height"], img["width"])

            if self.data_split == "train":
                if '20200709-subject-01' not in img_path and '20200813-subject-02' not in img_path:
                    continue
                
                num_path = img_path[:-16]
                syn_img_path = os.path.join(num_path, img_path[-10:-4] + '.png')
                
                if not os.path.exists(syn_img_path):
                    continue

                img_path = syn_img_path
                
                joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
                joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
                hand_type = ann["hand_type"]

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=2)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)

                if bbox is None:
                    continue

                bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
                bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
                bbox_size_x = bbox[2] / img["width"]
                bbox_size_y = bbox[3] / img["height"]
                bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size_x, bbox_size_y])

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "bbox": bbox,
                    "bbox_pos": bbox_pos,
                    "cam_param": cam_param,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "hand_type": hand_type,
                }
            else:
                joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)
                root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
                hand_type = ann["hand_type"]

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=2)
                bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)

                if bbox is None:
                    bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}

                mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
                mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

                data = {
                    "img_path": img_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "root_joint_cam": root_joint_cam,
                    "bbox": bbox,
                    "cam_param": cam_param,
                    "image_id": image_id,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "hand_type": hand_type,
                }

            datalist.append(data)
            
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_param = data["img_path"], data["img_shape"], data["bbox"], data["cam_param"]
        hand_type = data["hand_type"]
        do_flip = (hand_type == "left")

        # img
        img = load_img(img_path)
        original_img = copy.deepcopy(img)
        
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.input_img_shape, rot_prob=0, do_flip=do_flip, img_path=img_path)

        original_img = self.transform(original_img.astype(np.float32))
        img = self.transform(img.astype(np.float32))

        if self.data_split == "train":
            # 2D joint coordinate
            joints_img = data["joints_coord_img"]
            if do_flip:
                joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= self.input_img_shape[1]
            joints_img[:, 1] /= self.input_img_shape[0]

            # 3D joint camera coordinate
            joints_coord_cam = data["joints_coord_cam"]
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            if do_flip:
                joints_coord_cam[:, 0] *= -1

            # 3D data rotation augmentation
            rot_aug_mat = np.array(
                [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            # mano parameter
            mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            if do_flip:
                mano_pose[:, 1:] *= -1
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)
            bbox_pos = data["bbox_pos"]

            input = {
                "original_img": original_img,
                "img": img,
                "bbox_pos": bbox_pos,
                "joints_img": joints_img,
                "img2bb_trans": img2bb_trans,
                "rot_aug_mat": rot_aug_mat,
                "joints_coord_cam": joints_coord_cam,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "root_joint_cam": root_joint_cam,
                "cam_focal": cam_param["focal"],
                "cam_princpt": cam_param["princpt"],
            }

        else:
            root_joint_cam = data["root_joint_cam"]
            joints_coord_cam = data["joints_coord_cam"]

            # mano parameter
            mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

            # Only for rotation metric
            val_mano_pose = copy.deepcopy(mano_pose).reshape(-1, 3)
            
            if do_flip:
                val_mano_pose[:, 1:] *= -1
                joints_coord_cam[:, 0] *= -1 # root relative
            val_mano_pose = val_mano_pose.reshape(-1)

            # 2D joint coordinate
            joints_img = data["joints_coord_img"]
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= self.input_img_shape[1]
            joints_img[:, 1] /= self.input_img_shape[0]

            input = {
                "original_img": original_img,
                "img": img,
                "joints_img": joints_img,
                "img2bb_trans": img2bb_trans,
                "joints_coord_cam": joints_coord_cam,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "val_mano_pose": val_mano_pose,
                "root_joint_cam": root_joint_cam,
                "cam_focal": cam_param["focal"],
                "cam_princpt": cam_param["princpt"],
                "do_flip": np.array(do_flip).astype(int),
            }

        return input
