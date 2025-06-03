import os
import torch
import cv2
import numpy as np
import json
from PIL import Image
from utils import json_load

class VAE_Pose:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.name = "vae_pose"
        self.split = split
        self.load_dataset()

    # Annotations
    def load_dataset(self):
        if self.set_name == 'evaluation':
            dataset_name = 'evaluation'
        else:
            dataset_name = 'training'

        self.joint_list = json_load(os.path.join(self.base_path, 'joints.json'))
        self.verts_list = json_load(os.path.join(self.base_path, 'verts.json'))  
        
        image_names = []
        mask_names = []
        
        img_root = os.path.join(self.base_path, 'rgb')
        
        for imgname in sorted(os.listdir(img_root)):
            image_path = os.path.join(img_root, imgname)
            image_names.append(image_path)

        self.image_names = image_names
        
        del image_names

    def get_img(self, idx, test_on_normal):
        image_path = self.image_names[idx]
        img = Image.open(image_path).convert('RGB')
        
        if test_on_normal:
            img = img.resize((224, 224))
        
        return img

    def get_K(self, idx):
        K = np.array(self.K_list[idx])
        return K
    
    def get_scale(self, idx):
        scale = self.scale_list[idx]
        return scale
    
    def get_mano(self,idx):
        mano = self.mano_list[idx]
        mano = torch.FloatTensor(mano)
        return mano
    
    def get_joint(self,idx):
        joint = self.joint_list[idx]
        joint = torch.FloatTensor(joint)
        
        return joint

    def get_nimble_joint(self, idx):
        nimble_joint = self.nimble_joint_list[idx]
        nimble_joint = torch.FloatTensor(nimble_joint)
        
        return nimble_joint
    
    def get_vert(self, idx):
        verts = self.verts_list[idx]
        verts = torch.FloatTensor(verts)
        return verts
    
    def get_open_2dj(self, idx):
        open_2dj = self.open_2dj_list[idx]
        open_2dj = torch.FloatTensor(open_2dj)
        return open_2dj
    
    def get_open_2dj_con(self, idx):
        open_2dj_con = self.open_2dj_con_list[idx]
        open_2dj_con = torch.FloatTensor(open_2dj_con)
        return open_2dj_con
    def __len__(self):
        return len(self.image_names)
