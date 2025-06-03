import traceback
import random

import os
import torch
from torch.utils.data import Dataset, Subset

import numpy as np
import json
import warnings

from PIL import Image, ImageFilter
from torchvision.transforms import functional as func_transforms

import handutils
from utils import proj_func, intrinsic_matrix
from data_loaders.dex_ycb import DEX_YCB_SF
from data_loaders.freihand import FreiHand
from data_loaders.obman_syn import Obman_Syn
from data_loaders.random_nimble import Random_Nimble
from data_loaders.vae_pose import VAE_Pose

def get_dataset(
    dat_name,
    set_name,
    base_path,
    queries,
    limit_size=None,
    train = False,
    if_use_j2d: bool = False,
    syn_rgb_count: int = 10,
):
    if dat_name == "FreiHand":
        pose_dataset = FreiHand(
            base_path=base_path,
            set_name = set_name,
            syn_rgb_count = syn_rgb_count,
        )
        sides = 'right',
        sides = 'right',
    elif dat_name =='Dex':
        if set_name == 'training':
            set_name = 'train'
        else:
            set_name = 'test'
        pose_dataset = DEX_YCB_SF(
            base_path=base_path,
            data_split = set_name
        )
        sides = 'right'

    elif dat_name == 'Obman_Syn':
        pose_dataset = Obman_Syn(
            base_path=base_path,
            set_name = set_name,
            split = 'train',
            mode = 'all')
        sides = 'right',

    elif dat_name == 'Random_Nimble':
        pose_dataset = Random_Nimble(
            base_path=base_path,
            set_name = set_name,
        )
        sides = 'right',
    
    elif dat_name == 'VAE_Pose':
        pose_dataset = VAE_Pose(
            base_path=base_path,
            set_name = set_name,
        )
        sides = 'right',
    else:
        print("not supported dataset.")
        return
    
    dataset = HandDataset(
        dat_name,
        pose_dataset,
        queries=queries,
        sides = sides,
        is_train=train,
        if_use_j2d = if_use_j2d,
    )
    
    if limit_size is not None:
        if len(dataset) < limit_size:
            warnings.warn(
                "limit size {} > dataset size {}, working with full dataset".format(
                    limit_size, len(dataset)
                )
            )
        else:
            print( "Working wth subset of {} of size {}".format(dat_name, limit_size))
            dataset = Subset(dataset, list(range(limit_size)))
    return dataset

class HandDataset(Dataset):
    def __init__(
        self, 
        dat_name,
        pose_dataset, 
        if_use_j2d: bool,
        is_train=None, 
        queries = None,
        sides="both",
        center_idx=9,
    ):
        self.pose_dataset = pose_dataset
        
        self.dat_name = dat_name
        self.queries = queries
        
        self.sides = sides
        self.train = is_train
        self.inp_res = 320
        self.inp_res1 = 224
        self.center_idx = center_idx
        self.center_jittering = 0.2
        self.scale_jittering = 0.3
        self.max_rot = 0
        self.blur_radius = 0.5
        self.brightness = 0.3
        self.saturation = 0.3
        self.hue = 0.15
        self.contrast = 0.5
        self.block_rot = False
        self.black_padding = False
        self.data_pre = False
        self.if_use_j2d = if_use_j2d
    
    def __len__(self):
        return len(self.pose_dataset)
    
    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}

        if self.dat_name == 'FreiHand':
            image = self.pose_dataset.get_img(idx, self.test_on_normal)
            if 'images' in query:
                sample['images']=func_transforms.to_tensor(image).float()
            
            if 'bgimgs' in query:
                bgimg = self.pose_dataset.get_bgimg()
                sample['bgimgs']=func_transforms.to_tensor(bgimg).float()
            
            if 'refhand' in query:
                refhand = self.pose_dataset.get_refhand()
                sample['refhand']=func_transforms.to_tensor(refhand).float()

            if 'maskRGBs' in query:
                maskRGB = self.pose_dataset.get_maskRGB(idx)
                sample['maskRGBs']=maskRGB
            
            K = self.pose_dataset.get_K(idx)
            if 'Ks' in query:
                sample['Ks']=torch.FloatTensor(K)#K
            if 'scales' in query:
                scale = self.pose_dataset.get_scale(idx)
                sample['scales']=scale
            if 'manos' in query:
                mano = self.pose_dataset.get_mano(idx)
                sample['manos']=mano
            if 'joints' in query or 'trans_joints' in query:
                joint = self.pose_dataset.get_joint(idx, self.test_on_normal)
                if 'joints' in query:
                    sample['joints']=joint
            if 'nimble_joints' in query:
                nimble_joint = self.pose_dataset.get_nimble_joint(idx)
                sample['nimble_joints'] = nimble_joint
            if 'verts' in query or 'trans_verts' in query:
                verts = self.pose_dataset.get_vert(idx)
                if 'verts' in query:
                    sample['verts']=verts
            if 'open_2dj' in query or 'trans_open_2dj' in query:
                open_2dj = self.pose_dataset.get_open_2dj(idx)
                if 'open_2dj' in query:
                    sample['open_2dj']=open_2dj
                open_2dj_con = self.pose_dataset.get_open_2dj_con(idx)
                sample['open_2dj_con']=open_2dj_con
            if 'cv_images' in query:
                cv_image = self.pose_dataset.get_cv_image(idx)
                sample['cv_images']=cv_image
            sample['idxs']=idx
            if 'masks' in query or 'trans_masks' in query:
                if idx >= 32560:
                    idx_this = idx % 32560
                else:
                    idx_this = idx
                mask = self.pose_dataset.get_mask(idx_this)
                if 'masks' in query:
                    sample['masks']=torch.round(func_transforms.to_tensor(mask))
            if 'CRFmasks' in query:
                if idx >= 32560:
                    idx_this = idx % 32560
                else:
                    idx_this = idx
                CRFmask = self.pose_dataset.get_CRFmask(idx_this)
                sample['CRFmasks']=torch.round(func_transforms.to_tensor(CRFmask))

            if self.train:
                if 'trans_images' in query:
                    center = np.asarray([112, 112])
                    scale = 224

                    # Random rotations
                    rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
                    rot_mat = np.array(
                        [
                            [np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1],
                        ]
                    ).astype(np.float32)
                    affinetrans, post_rot_trans = handutils.get_affine_transform(
                        center, scale, [224, 224], rot=rot
                    )
                    trans_images = handutils.transform_img(
                        image, affinetrans, [224, 224]
                    )
                    
                    trans_images = func_transforms.to_tensor(trans_images).float()
                    sample['trans_images'] = trans_images
                    sample['post_rot_trans'] = post_rot_trans
                    sample['rot'] = rot
        
                if 'trans_open_2dj' in query:
                    trans_open_j2d = handutils.transform_coords(open_2dj.numpy(),affinetrans)
                    sample['trans_open_2dj']=torch.from_numpy(np.array(trans_open_j2d)).float()
                if 'trans_Ks' in query:
                    trans_Ks = post_rot_trans.dot(K)
                    sample['trans_Ks']=torch.from_numpy(trans_Ks).float()
                if 'trans_CRFmasks' in query:
                    trans_CRFmasks = handutils.transform_img(
                        CRFmask, affinetrans, [224, 224]
                    )
                    sample['trans_CRFmasks']=torch.round(func_transforms.to_tensor(trans_CRFmasks))
                if 'trans_masks' in query:
                    trans_masks = handutils.transform_img(
                        mask, affinetrans, [224, 224]
                    )
                    sample['trans_masks']=torch.round(func_transforms.to_tensor(trans_masks))
                if 'trans_joints' in query:
                    trans_joint = rot_mat.dot(
                        joint.transpose(1, 0)
                    ).transpose()
                    sample['trans_joints'] = torch.from_numpy(trans_joint)
                if 'trans_verts' in query:
                    trans_verts = rot_mat.dot(
                        verts.transpose(1, 0)
                    ).transpose()
                    sample['trans_verts'] = torch.from_numpy(trans_verts)
                    
            if self.if_use_j2d:
                if 'images' in sample:
                    assert 'open_2dj' in sample, "You should include 'open_2dj' in queries to use it as input."
                    sample['images'] = torch.cat([sample['images'], sample['open_2dj']], dim=0)
                if 'trans_images' in sample:
                    assert 'trans_open_2dj' in sample, "You should include 'trans_open_2dj' in queries to use it as input."
                    print(sample['trans_images'].shape, sample['trans_open_2dj'].shape )
                    sample['trans_images'] = torch.cat([sample['trans_images'], sample['trans_open_2dj']], dim=0)

        if self.dat_name == 'Dex':
            dex_sample = self.pose_dataset.__getitem__(idx)

            if 'images' in query or 'trans_images' in query:
                if self.train:
                    img_array = np.transpose((dex_sample['img'].numpy()).astype(np.uint8), (1, 2, 0))
                    image = Image.fromarray(img_array)
                    sample['images'] = func_transforms.to_tensor(image).float() # torch.FloatTensor
                else:
                    sample['images'] = dex_sample['img'].numpy() / 255. # torch.FloatTensor
                
                sample['ori_images'] = dex_sample['original_img']

            if 'Ks' in query or 'trans_Ks' in query:
                K = intrinsic_matrix(dex_sample['cam_focal'][0], dex_sample['cam_focal'][1], dex_sample['cam_princpt'][0], dex_sample['cam_princpt'][1])
                sample['Ks']=torch.FloatTensor(K)
                
            if 'joints' in query or 'trans_joints' in query:
                sample['joints'] = dex_sample['joints_coord_cam']
                sample['j2d_gt'] = dex_sample['joints_img']
                
            if 'verts' in query or 'trans_verts' in query:
                sample['verts'] = dex_sample['verts']
            sample['idxs']=idx
        
        ### Obman
        if self.dat_name == 'Obman_Syn':
            self.inp_res = 256
            if 'sides' in query:
                hand_side = self.pose_dataset.get_sides(idx)
                # Flip if needed
                if self.sides == "right" and hand_side == "left":
                    flip = True
                    hand_side = "right"
                elif self.sides == "left" and hand_side == "right":
                    flip = True
                    hand_side = "left"
                else:
                    flip = False
                sample['sides'] = hand_side
            else:
                flip = False

            # Get original image
            if 'base_images' in query or 'trans_images' in query:
                center, scale = self.pose_dataset.get_center_scale(idx)
                needs_center_scale = True
                img = self.pose_dataset.get_img(idx)
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if 'base_images' in query:
                    sample['base_images'] = func_transforms.to_tensor(img)#cyj
            else:
                needs_center_scale = False
            
            # Flip and image 2d if needed
            if flip:
                center[0] = img.size[0] - center[0]
            
            # Data augmentation
            if self.train and needs_center_scale:
                # Randomly jitter center
                # Center is located in square of size 2*center_jitter_factor
                # in center of cropped image
                center_offsets = (
                    self.center_jittering
                    * scale
                    * np.random.uniform(low=-1, high=1, size=2)
                )
                center = center + center_offsets.astype(int)

                # Scale jittering
                scale_jittering = self.scale_jittering * np.random.randn() + 1
                scale_jittering = np.clip(
                    scale_jittering,
                    1 - self.scale_jittering,
                    1 + self.scale_jittering,
                )
                scale = scale * scale_jittering

                rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
            else:
                rot = 0
            if self.block_rot:
                rot = self.max_rot
            rot_mat = np.array(
                [
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1],
                ]
            ).astype(np.float32)

            # Get 2D hand joints
            if ('trans_j2d' in query) or ('trans_images' in query):
                affinetrans, post_rot_trans = handutils.get_affine_transform(
                    center, scale, [self.inp_res, self.inp_res], rot=rot
                )
            if ('base_j2d' in query) or ('trans_j2d' in query):
                joints2d = self.pose_dataset.get_joints2d(idx)
                if flip:
                    joints2d = joints2d.copy()
                    joints2d[:, 0] = img.size[0] - joints2d[:, 0]
                if 'base_j2d' in query:
                    sample['base_j2d'] = torch.from_numpy(joints2d)
            if 'trans_j2d' in query:
                rows = handutils.transform_coords(joints2d, affinetrans)
                #sample['trans_j2d0'] = torch.from_numpy(np.array(rows))
                trans_joints2d= torch.from_numpy(np.array(rows)).float()
                
                x_center_move = -(self.inp_res - self.inp_res1)/2
                y_center_move = -(self.inp_res - self.inp_res1)/2
                crop_x_max = (self.inp_res + self.inp_res1)/2
                crop_x_min = (self.inp_res - self.inp_res1)/2
                crop_y_max = (self.inp_res + self.inp_res1)/2
                crop_y_min = (self.inp_res - self.inp_res1)/2
                #sample['x_center_move']=x_center_move
                #sample['y_center_move']=y_center_move
                trans_joints2d = trans_joints2d + torch.tensor([x_center_move, y_center_move]).float()
                sample['trans_j2d'] = trans_joints2d
                #crop_size_best = 2*torch.max(max_uv-center,center-min_uv)
                #crop_size_best = torch.max(crop_size_best)
                #crop_size_best = torch.min(torch.max(crop_size_best,torch.ones(1)*50.0),torch.ones(1)*500.0)
                '''
                '''

            if 'base_camintrs' in query or 'trans_camintrs' in query:
                camintr = self.pose_dataset.get_camintr(idx)
                if 'base_camintrs' in query:
                    sample['base_camintrs'] = camintr
                    sample['Ks'] = camintr
                    sample['Ks'] = camintr
                if 'trans_camintrs' in query:
                    # Rotation is applied as extr transform
                    new_camintr = post_rot_trans.dot(camintr)
                    # crop is applied as a tranform

                    #sample['trans_camintrs'] = torch.from_numpy(new_camintr)
                    trans_camintrs = torch.from_numpy(new_camintr)
                    trans_camintrs = trans_camintrs + torch.tensor([[0,0,x_center_move],[0,0,y_center_move],[0,0,0]]).float()
                    sample['trans_camintrs'] = trans_camintrs
                    sample['Ks'] = trans_camintrs
            
            # Get segmentation
            if 'base_segms' in query or 'trans_segms' in query:
                segm = self.pose_dataset.get_segm(idx)#[256, 256]
                #print("segm:",segm.size)
                if flip:
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                if 'base_segms' in query:
                    #trans = transforms.ToTensor()
                    #segm = trans(segm)
                    sample['base_segms'] = func_transforms.to_tensor(segm)#cyj
                if 'trans_segms' in query:
                    segm = handutils.transform_img(
                        segm, affinetrans, [self.inp_res, self.inp_res]
                    )
                    #segm = segm.crop((0, 0, self.inp_res1, self.inp_res1))
                    segm = segm.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                    segm = func_transforms.to_tensor(segm)
                    sample['trans_segms'] = segm

            # Get instance
            if ('trans_instances' in query) or ('base_instances' in query):
                segm_img1, segm_img2 = self.pose_dataset.get_instance(idx)
                #print("instance_img:",segm_img1.size)
                if flip:
                    segm_img1 = segm_img1.transpose(Image.FLIP_LEFT_RIGHT)
                    segm_img2 = segm_img2.transpose(Image.FLIP_LEFT_RIGHT)

                indices1 = torch.LongTensor([0])#hand
                indices2 = torch.LongTensor([1,2])#object zero-like

                if 'base_instances' in query:
                    img1_slice = torch.index_select(func_transforms.to_tensor(segm_img1), 0, indices1)
                    img2_slice = torch.index_select(func_transforms.to_tensor(segm_img2), 0, indices2)
                    instance = torch.cat((img1_slice,img2_slice),0)
                    sample['base_instances'] = instance

                segm_img1 = handutils.transform_img(
                    segm_img1, affinetrans, [self.inp_res, self.inp_res]
                )
                #segm_img1 = segm_img1.crop((0, 0, self.inp_res1, self.inp_res1))
                segm_img1 = segm_img1.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                segm_img1 = func_transforms.to_tensor(segm_img1)

                segm_img2 = handutils.transform_img(
                    segm_img2, affinetrans, [self.inp_res, self.inp_res]
                )
                #segm_img2 = segm_img2.crop((0, 0, self.inp_res1, self.inp_res1))
                segm_img2 = segm_img2.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                segm_img2 = func_transforms.to_tensor(segm_img2)

                

                img1_slice = torch.index_select(segm_img1, 0, indices1)
                img2_slice = torch.index_select(segm_img2, 0, indices2)

                instance = torch.cat((img1_slice,img2_slice),0)
                
                sample['trans_instances'] = instance

            # Get 3d joints
            if ('base_j3d' in query) or ('trans_j3d' in query):
                joints3d = self.pose_dataset.get_joints3d(idx)
                if flip:
                    joints3d[:, 0] = -joints3d[:, 0]

                if 'base_j3d' in query:
                    sample['base_j3d'] = joints3d
                if self.train:
                    joints3d = rot_mat.dot(
                        joints3d.transpose(1, 0)
                    ).transpose()
                # Compute 3D center
                '''
                if self.center_idx is not None:
                    if self.center_idx == -1:
                        center3d = (joints3d[9] + joints3d[0]) / 2
                    else:
                        center3d = joints3d[self.center_idx]
                if 'trans_j3d' in query and (
                    self.center_idx is not None
                ):
                    joints3d = joints3d - center3d
                '''
                if 'trans_j3d' in query:
                    sample['trans_j3d'] = torch.from_numpy(
                        joints3d
                    )
                    sample['joints'] = sample['trans_j3d']

            # get verts
            if 'base_verts' in query or ('trans_verts' in query):
                hand_verts3d = self.pose_dataset.get_verts3d(idx)
                if flip:
                    hand_verts3d[:, 0] = -hand_verts3d[:, 0]
                sample['base_verts'] = torch.from_numpy(hand_verts3d)
                hand_verts3d = rot_mat.dot(
                    hand_verts3d.transpose(1, 0)
                ).transpose()
                '''
                if self.center_idx is not None:
                    hand_verts3d = hand_verts3d - center3d
                '''
                sample['trans_verts'] = torch.from_numpy(hand_verts3d)

            # Get rgb image
            if 'trans_images' in query:
                # Data augmentation
                if self.train:
                    blur_radius = random.random() * self.blur_radius
                    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                
                # Transform and crop
                img = handutils.transform_img(
                    img, affinetrans, [self.inp_res, self.inp_res]
                )
                #img = img.crop((0, 0, self.inp_res1, self.inp_res1))
                img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                #img = img.crop(((self.inp_res-self.inp_res1)/2, (self.inp_res-self.inp_res1)/2,  (self.inp_res+self.inp_res1)/2, (self.inp_res+self.inp_res1)/2))
                # Tensorize and normalize_img
                img = func_transforms.to_tensor(img).float()
                if self.black_padding:
                    padding_ratio = 0.2
                    padding_size = int(self.inp_res * padding_ratio)
                    img[:, 0:padding_size, :] = 0
                    img[:, -padding_size:-1, :] = 0
                    img[:, :, 0:padding_size] = 0
                    img[:, :, -padding_size:-1] = 0

                if self.data_pre:
                    if self.normalize_img:
                        img = func_transforms.normalize(img, self.mean, self.std)
                    else:
                        img = func_transforms.normalize(
                            img, [0.5, 0.5, 0.5], [1, 1, 1]
                        )
                
                sample['trans_images'] = img

            # if 'center3d' in query:
            #     sample['center3d'] = torch.from_numpy(center3d)
            
            sample['idxs'] = idx
        
        if self.dat_name == 'VAE_Pose' or self.dat_name == 'Random_Nimble':
            image = self.pose_dataset.get_img(idx, self.test_on_normal)
            if 'images' in query:
                sample['images']=func_transforms.to_tensor(image).float()#image
            
            K = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
            ])
            if 'Ks' in query:
                #K_0 = torch.zeros(3,1)
                #K = torch.cat((K,K_0),dim=1).float()
                sample['Ks']=torch.FloatTensor(K) #K
            if 'joints' in query or 'trans_joints' in query:
                joint = self.pose_dataset.get_joint(idx, self.test_on_normal)
                if 'joints' in query:
                    sample['joints']=joint
            if 'verts' in query or 'trans_verts' in query:
                verts = self.pose_dataset.get_vert(idx)
                if 'verts' in query:
                    sample['verts']=verts
            if 'cv_images' in query:
                cv_image = self.pose_dataset.get_cv_image(idx)
                sample['cv_images']=cv_image
            sample['idxs']=idx

            # augmentated results
            if self.train:
                if 'trans_images' in query:
                    center = np.asarray([112, 112])
                    scale = 224

                    # Random rotations
                    rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
                    rot_mat = np.array(
                        [
                            [np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1],
                        ]
                    ).astype(np.float32)
                    affinetrans, post_rot_trans = handutils.get_affine_transform(
                        center, scale, [224, 224], rot=rot
                    )
                    trans_images = handutils.transform_img(
                        image, affinetrans, [224, 224]
                    )
                    
                    trans_images = func_transforms.to_tensor(trans_images).float()
                    
                    # sample['trans_images'] = func_transforms.to_tensor(trans_images).float()
                    sample['trans_images'] = trans_images
                    sample['post_rot_trans']=post_rot_trans
                    sample['rot'] = rot
        
                if 'trans_open_2dj' in query:
                    trans_open_j2d = handutils.transform_coords(open_2dj.numpy(),affinetrans)
                    sample['trans_open_2dj']=torch.from_numpy(np.array(trans_open_j2d)).float()
                if 'trans_Ks' in query:
                    trans_Ks = post_rot_trans.dot(K)
                    sample['trans_Ks']=torch.from_numpy(trans_Ks).float()
                if 'trans_CRFmasks' in query:
                    trans_CRFmasks = handutils.transform_img(
                        CRFmask, affinetrans, [224, 224]
                    )
                    sample['trans_CRFmasks']=torch.round(func_transforms.to_tensor(trans_CRFmasks))
                if 'trans_joints' in query:
                    trans_joint = rot_mat.dot(
                        joint.transpose(1, 0)
                    ).transpose()
                    sample['trans_joints'] = torch.from_numpy(trans_joint)
                if 'trans_verts' in query:
                    trans_verts = rot_mat.dot(
                        verts.transpose(1, 0)
                    ).transpose()
                    sample['trans_verts'] = torch.from_numpy(trans_verts)
                #sample['rot_mat'] = torch.from_numpy(rot_mat)
            if self.if_use_j2d:
                if 'images' in sample:
                    assert 'open_2dj' in sample, "You should include 'open_2dj' in queries to use it as input."
                    sample['images'] = torch.cat([sample['images'], sample['open_2dj']], dim=0)
                if 'trans_images' in sample:
                    assert 'trans_open_2dj' in sample, "You should include 'trans_open_2dj' in queries to use it as input."
                    print(sample['trans_images'].shape, sample['trans_open_2dj'].shape )
                    sample['trans_images'] = torch.cat([sample['trans_images'], sample['trans_open_2dj']], dim=0)
            

        return sample
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            random_idx = random.randint(0, len(self))
            print("Encountered error processing sample {}, try to use a random idx {} instead".format(idx, random_idx))
            sample = self.get_sample(random_idx, self.queries)
        return sample

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map 
