import os
import torch
import numpy as np
from PIL import Image
from random import choice
from torchvision.transforms import functional as func_transforms
from utils import json_load

class FreiHand:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
        syn_rgb_count = 10,
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.syn_rgb_count = syn_rgb_count
        
        self.load_dataset()
        self.name = "FreiHand"
        self.split = split

    # Annotations
    def load_dataset(self):
        if self.set_name == 'evaluation':
            dataset_name = 'evaluation'
        else:
            dataset_name = 'training'

        self.K_list = json_load(os.path.join(self.base_path, 'FreiHand', '%s_K.json' % dataset_name))
        self.scale_list = json_load(os.path.join(self.base_path, 'FreiHand', '%s_scale.json' % dataset_name))
        self.mano_list = json_load(os.path.join(self.base_path, 'FreiHand', '%s_mano.json' % dataset_name))
        self.joint_list = json_load(os.path.join(self.base_path, 'FreiHand', '%s_xyz.json' % dataset_name))
        self.verts_list = json_load(os.path.join(self.base_path, 'FreiHand', '%s_verts.json' % self.set_name))
                
        if self.set_name == 'training' or self.set_name == 'trainval_train' or self.set_name == 'trainval_val':
            mask_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, 'FreiHand', dataset_name, 'mask')))]
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in mask_idxs]

            self.K_list = self.K_list * self.syn_rgb_count
            self.scale_list = self.scale_list * self.syn_rgb_count
            self.mano_list = self.mano_list * self.syn_rgb_count
            self.joint_list = self.joint_list * self.syn_rgb_count
            self.verts_list = self.verts_list * self.syn_rgb_count
            
        elif self.set_name == 'evaluation':
            img_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, 'FreiHand', self.set_name, 'rgb'))) if imgname.endswith('.jpg')]
            
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in img_idxs]    
        
        image_names = []
        mask_names = []
        
        if self.set_name == 'training':
            syn_rgb_count = self.syn_rgb_count
            
            for count in range(syn_rgb_count):
                for idx, prefix in enumerate(prefixes):
                    mask_path = os.path.join(self.base_path, 'segmentation', '{}.png'.format(prefix))
                    mask_names.append(mask_path)
                    
                    image_path = os.path.join(self.base_path, 'rgb' + '_' + str(count), '{}.png'.format(prefix))
                    image_names.append(image_path)
            
            del mask_idxs

        elif self.set_name == 'evaluation':
            for idx, prefix in enumerate(prefixes):
                image_path = os.path.join(self.base_path, 'FreiHand', dataset_name, 'rgb', '{}.jpg'.format(prefix))
                image_names.append(image_path)
        self.image_names = image_names
        self.mask_names = mask_names
        
        del image_names
        del prefixes

    def get_img(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path).convert('RGB')
        
        return img
    
    def get_mask(self, idx):
        mask_path = self.mask_names[idx]
        mask = Image.open(mask_path)

        return mask

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

    def __len__(self):
        return len(self.image_names)
