import os
import torch
import cv2
import pickle
import numpy as np
from PIL import Image
import handutils

class Obman_Syn:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split='train',
        joint_nb=21,
        mini_factor=None,
        mode = 'all',
    ):
        self.name = "obman"
        self.set_name = set_name
        self.base_path = base_path
        self.split = split
        self.syn_img_root = './syn_occ/rgb' # syn_occ rgb images root

        self.segment = False
        self.root_palm = False
        self.segmented_depth = True
        self.cam_intr = np.array(
            [[480.0, 0.0, 128.0], [0.0, 480.0, 128.0], [0.0, 0.0, 1.0]]
        ).astype(np.float32)
        self.mini_factor = mini_factor
        self.cam_extr = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ).astype(np.float32)
        self.joint_nb = joint_nb
        obman_root = os.path.join(base_path, split)
        self.segm_folder = os.path.join(obman_root, "segm")
        self.prefix_template = "{:08d}"
        self.meta_folder = os.path.join(obman_root, "meta")
        self.meta_folder_list = sorted(os.listdir(self.meta_folder))
        self.coord2d_folder = os.path.join(obman_root, "coords2d")

        self.mode = mode
        if self.mode == 'hand':
            self.rgb_folder = os.path.join(obman_root, "rgb_hand")
        elif self.mode == 'all':
            self.rgb_folder = os.path.join(obman_root, "rgb")
        self.load_dataset()
    
    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, "{}.png".format(prefix))
        return image_path
    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path
    # Annotations
    def load_dataset(self):
        idxs = [
                int(imgname.split(".")[0])
                for imgname in sorted(os.listdir(self.meta_folder))
            ]
        if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]

        prefixes = [self.prefix_template.format(idx) for idx in idxs]
        print(
            "Got {} samples for split {}, generating cache!".format(
                len(idxs), self.split
            )
        )

        image_names = []
        all_joints2d = []
        all_joints3d = []
        hand_sides = []
        hand_poses = []
        hand_pcas = []
        hand_verts3d = []
        depth_infos = []
        
        for idx, prefix in enumerate(prefixes):
            if idx > 60000:
                break
            
            meta_path = os.path.join(
                self.meta_folder, "{}.pkl".format(prefix)
            )
            with open(meta_path, "rb") as meta_f:
                meta_info = pickle.load(meta_f)
            image_path = self._get_image_path(prefix)
            
            syn_img_path = os.path.join(self.syn_img_root, str(idx).rjust(8, '0') + '.png')

            if os.path.exists(syn_img_path) == False:
                continue
            
            image_path = syn_img_path
            
            image_names.append(image_path)
            all_joints2d.append(meta_info["coords_2d"])
            all_joints3d.append(meta_info["coords_3d"])
            hand_verts3d.append(meta_info["verts_3d"])
            hand_sides.append(meta_info["side"])
            hand_poses.append(meta_info["hand_pose"])
            hand_pcas.append(meta_info["pca_pose"])
            depth_infos.append(
                {
                    "depth_min": meta_info["depth_min"],
                    "depth_max": meta_info["depth_max"],
                    "hand_depth_min": meta_info["hand_depth_min"],
                    "hand_depth_max": meta_info["hand_depth_max"],
                    "obj_depth_min": meta_info["obj_depth_min"],
                    "obj_depth_max": meta_info["obj_depth_max"],
                }
            )   

            annotations = {
                "depth_infos": depth_infos,
                "image_names": image_names,
                "joints2d": all_joints2d,
                "joints3d": all_joints3d,
                "hand_sides": hand_sides,
                "hand_poses": hand_poses,
                "hand_pcas": hand_pcas,
                "hand_verts3d": hand_verts3d,
            }
            
        # Set dataset attributes
        selected_idxs = list(range(len(image_names)))
        image_names = [
            annotations["image_names"][idx] for idx in selected_idxs
        ]
        joints3d = [annotations["joints3d"][idx] for idx in selected_idxs]
        joints2d = [annotations["joints2d"][idx] for idx in selected_idxs]
        hand_sides = [annotations["hand_sides"][idx] for idx in selected_idxs]
        hand_pcas = [annotations["hand_pcas"][idx] for idx in selected_idxs]
        hand_verts3d = [
            annotations["hand_verts3d"][idx] for idx in selected_idxs
        ]
        
        if "depth_infos" in annotations:
            has_depth_info = True
            depth_infos = [
                annotations["depth_infos"][idx] for idx in selected_idxs
            ]
        else:
            has_depth_info = False
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.hand_verts3d = hand_verts3d
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}
    
    def get_center_scale(self, idx, scale_factor=2.2):#scale_factor=2.2  2.5
        joints2d = self.get_joints2d(idx)
        center = handutils.get_annot_center(joints2d)
        scale = handutils.get_annot_scale(
            joints2d, scale_factor=scale_factor
        )
        return center, scale

    def get_img(self, idx):
        image_path = self.image_names[idx]
        side = self.get_sides(idx)
        if self.segment:
            if self.mode == "all":
                segm_path = image_path.replace("rgb", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "hand":
                segm_path = image_path.replace("rgb_hand", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "obj":
                segm_path = image_path.replace("rgb_obj", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "ho":
                segm_path = image_path.replace("rgb", "segm").replace(
                    "jpg", "png"
                )
                segm_path1 = image_path.replace("rgb_hand", "segm").replace(
                    "jpg", "png"
                )
                segm_path2 = image_path.replace("rgb_obj", "segm").replace(
                    "jpg", "png"
                )
            
            img = cv2.imread(image_path, 1)
            if img is None:
                raise ValueError("cv2 could not open {}".format(image_path))
            segm_img = cv2.imread(segm_path, 1)
            if segm_img is None:
                raise ValueError("cv2 could not open {}".format(segm_path))
            if self.mode == "all":
                segm_img = segm_img[:, :, 0]
            elif self.mode == "hand":
                segm_img = segm_img[:, :, 1]
            elif self.mode == "obj":
                segm_img = segm_img[:, :, 2]
            segm_img = _get_segm(segm_img, side=side)
            segm_img = segm_img.sum(2)[:, :, np.newaxis]
            # blacken not segmented
            img[~segm_img.astype(bool).repeat(3, 2)] = 0
            img = Image.fromarray(img[:, :, ::-1])
            return img
        else:
            img = Image.open(image_path)
            img = img.convert("RGB")
        return img

    def get_segm(self, idx, pil_image=True):
        side = self.get_sides(idx)
        image_path = self.image_names[idx]
        pkl_name = self.meta_folder_list[idx]
        image_path = os.path.join(obman_root, 'rgb', pkl_name.replace('pkl', 'jpg'))
        if self.mode == "all":
            image_path = image_path.replace("rgb", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "segm").replace(
                "jpg", "png"
            )

        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))
        if self.mode == "all":
            segm_img = _get_segm(img[:, :, 0], side=side)
        elif self.mode == "hand":
            segm_img = _get_segm(img[:, :, 1], side=side)
        elif self.mode == "obj":
            segm_img = _get_segm(img[:, :, 2], side=side)
        if pil_image:
            segm_img = Image.fromarray((255 * segm_img).astype(np.uint8))
        return segm_img
    
    def get_instance(self, idx, pil_image=True):
        side = self.get_sides(idx)
        image_path = self.image_names[idx]
        pkl_path = self.meta_folder_list[idx]
        image_path = os.path.join(obman_root, 'rgb', pkl_name.replace('pkl', 'jpg'))
        #print(image_path)
        if self.mode == "hand":
            image_path0 = image_path.replace("rgb_hand", "segm").replace(
                    "jpg", "png"
                )
        else:
            image_path0 = image_path.replace("rgb", "segm").replace("jpg", "png")

        img = cv2.imread(image_path0, 1)

        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path0))
        
        segm_img1 = _get_segm(img[:, :, 1], side=side)#hand
        segm_img2 = _get_segm(img[:, :, 2], side=side)#obj
        if pil_image:
            #segm_img = Image.fromarray((255 * segm_img).astype(np.uint8))
            segm_img1 = Image.fromarray((255 * segm_img1).astype(np.uint8))
            segm_img2 = Image.fromarray((255 * segm_img2).astype(np.uint8))
        #segm_img = 0
        return segm_img1, segm_img2

    def get_depth(self, idx):
        image_path = self.image_names[idx]
        pkl_path = self.meta_folder_list[idx]
        image_path = os.path.join(obman_root, 'rgb', pkl_name.replace('pkl', 'jpg'))
        if self.mode == "all":
            image_path = image_path.replace("rgb", "depth")
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "depth")
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "depth")
        image_path = image_path.replace("jpg", "png")

        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))

        depth_info = self.depth_infos[idx]
        if self.mode == "all":
            img = img[:, :, 0]
            depth_max = depth_info["depth_max"]
            depth_min = depth_info["depth_min"]
        elif self.mode == "hand":
            img = img[:, :, 1]
            depth_max = depth_info["hand_depth_max"]
            depth_min = depth_info["hand_depth_min"]
        elif self.mode == "obj":
            img = img[:, :, 2]
            depth_max = depth_info["obj_depth_max"]
            depth_min = depth_info["obj_depth_min"]
        assert (
            img.max() == 255
        ), "Max value of depth jpg should be 255, not {}".format(img.max())

        img = (img - 1) / 254 * (depth_min - depth_max) + depth_max
        if self.segmented_depth:
            obj_hand_segm = (np.asarray(self.get_segm(idx)) / 255).astype(
                int
            )
            segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
            img = img * segm
        img = Image.fromarray(img)
        
        return img

    def get_joints2d(self, idx):
        return self.joints2d[idx].astype(np.float32)

    def get_joints3d(self, idx):
        joints3d = self.joints3d[idx]
        if self.root_palm:
            # Replace wrist with palm
            verts3d = self.hand_verts3d[idx]
            palm = (verts3d[95] + verts3d[218]) / 2
            joints3d = np.concatenate([palm[np.newaxis, :], joints3d[1:]])
        # No hom coordinates needed because no translation
        assert (
            np.linalg.norm(self.cam_extr[:, 3]) == 0
        ), "extr camera should have no translation"

        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        #return 1000 * joints3d
        return joints3d

    def get_verts3d(self, idx):
        verts3d = self.hand_verts3d[idx]
        verts3d = self.cam_extr[:3, :3].dot(verts3d.transpose()).transpose()
        #return 1000 * verts3d
        return verts3d
    def get_sides(self, idx):
        return self.hand_sides[idx]

    def get_camintr(self, idx):
        return self.cam_intr

    def __len__(self):
        return len(self.image_names)

def _get_segm(img, side="left"):
    if side == "right":
        hand_segm_img = (img == 22).astype(float) + (img == 24).astype(float)
    elif side == "left":
        hand_segm_img = (img == 21).astype(float) + (img == 23).astype(float)
    else:
        raise ValueError("Got side {}, expected [right|left]".format(side))

    obj_segm_img = (img == 100).astype(float)
    segm_img = np.stack(
        [hand_segm_img, obj_segm_img, np.zeros_like(hand_segm_img)], axis=2
    )
    return segm_img
