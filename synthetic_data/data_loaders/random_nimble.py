import os
import torch
import cv2
import numpy as np
from random import choice
from PIL import Image
from utils import json_load
import skimage.io as io
from torchvision.transforms import functional as func_transforms

class Random_Nimble:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.name = "random_nimble"
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

    def get_cv_image(self, idx):
        image_path = self.image_names[idx]
        cv_image = cv2.imread(image_path)
        return cv_image
    
    def get_mask(self, idx):
        mask_path = self.mask_names[idx]
        mask = Image.open(mask_path)

        return mask
    
    def get_CRFmask(self, idx):
        CRFmask_path = os.path.join(self.CRFmask_dir,"{:08d}.png".format(idx))
        mask = Image.open(CRFmask_path)

        return mask
    
    def get_bgimg(self):
        not_ok_bg = True
        while not_ok_bg:
            bgimg_path = os.path.join(self.bgimgs_dir, choice(self.bgimgs_filename))
            bgimg = Image.open(bgimg_path)
            if len(bgimg.split())==3:
                not_ok_bg=False
        bgimg = func_transforms.resize(bgimg,(224,224))
        return bgimg

    def get_refhand(self):
        not_ok_ref = True
        while not_ok_ref:
            refhand = Image.open(choice(self.refhand_filename))
            if len(refhand.split())==3:
                not_ok_ref=False
        width, height = refhand.size
        if width == 640 and height == 480:#MVHP
            refhand = func_transforms.center_crop(refhand, [400,400])
            
        refhand = func_transforms.rotate(refhand, angle=np.random.randint(-180,180))
        
        refhand = func_transforms.resize(refhand,(224,224))
        return refhand

    def get_maskRGB(self, idx):
        
        image_path = self.image_names[idx]
        img = io.imread(image_path)
        if idx >= 32560:
            idx = idx % 32560
            image_path = self.image_names[idx]
        mask_img =io.imread(image_path.replace('rgb', 'mask'), 1)
        mask_img = np.rint(mask_img)
        img[~mask_img.astype(bool)] = 0
        img = func_transforms.to_tensor(img).float()
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
        joint = torch.FloatTensor(joint) / 1000.0
        
        return joint
    
    def get_nimble_joint(self, idx):
        nimble_joint = self.nimble_joint_list[idx]
        nimble_joint = torch.FloatTensor(nimble_joint)
        
        return nimble_joint
    
    def get_vert(self, idx):
        verts = self.verts_list[idx]
        verts = torch.FloatTensor(verts) / 1000.0
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
