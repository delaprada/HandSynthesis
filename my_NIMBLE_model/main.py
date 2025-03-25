import os
import torch
import json
import numpy as np
import random
from pytorch3d.structures.meshes import Meshes
from NIMBLELayer import NIMBLELayer
from utils import batch_to_tensor_device, save_textured_nimble, mano_v2j_reg, save_tex, smooth_mesh

def random_rotate(vertices):
    random.seed(None)
    random_angles = [random.uniform(0, 360) for _ in range(bn)]

    rotation_matrices = []

    for angle in random_angles:
        angle_x = angle
        angle_y = angle
        angle_z = angle

        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)
        angle_z_rad = np.radians(angle_z)

        rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                            [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])

        rotation_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                            [0, 1, 0],
                            [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

        rotation_z = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                            [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                            [0, 0, 1]])

        rotation_matrix = np.dot(np.dot(rotation_z, rotation_y), rotation_x)
        rotation_matrices.append(rotation_matrix.T)

    rotated_vertices = np.einsum('ijk,ikl->ijl', vertices, np.array(rotation_matrices, dtype='float32'))

    return rotated_vertices, rotation_matrices

if __name__ == "__main__":
    device = 'cpu'

    pm_dict_name = "./assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = "./assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg = None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    
    start = 0
    num_samples = 20
    save_obj = True

    output_folder = "../raw_data" # specify output folder
    os.makedirs(output_folder, exist_ok=True)
    
    step = 10
    
    for idx in range(start, start + num_samples, step):
        bn = step

        pose_param = torch.rand(bn, 30) * 2 - 1
        shape_param = torch.rand(bn, 20) * 2 - 1
        tex_param = torch.rand(bn, 10) * 2 - 1
        
        pose_param = pose_param.to(device)
        shape_param = shape_param.to(device)
        tex_param = tex_param.to(device)

        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
        tex_img = tex_img.detach().cpu().numpy()
        
        root_vertice = skin_v[:, 2078].unsqueeze(1) # rotation center

        skin_v -= root_vertice
        
        # random rotate to boost diversity
        skin_v, R = random_rotate(skin_v.detach().cpu().numpy())
        # skin_v = torch.Tensor(random_rotate(skin_v.detach().cpu().numpy())).to(device)
        skin_v = torch.Tensor(skin_v).to(device)
        skin_v += root_vertice

        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(bn, 1, 1))
        muscle_p3dmesh = Meshes(muscle_v, nlayer.muscle_f.repeat(bn, 1, 1))
        bone_p3dmesh = Meshes(bone_v, nlayer.bone_f.repeat(bn, 1, 1))
        
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)

        skin_mano_v = nlayer.nimble_to_mano(skin_v, is_surface=True)
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
            
        mano_joints = mano_v2j_reg(skin_mano_v.cpu()).detach().cpu().numpy()
        
        for i in range(bn):
            length = 8 # name length
            id = str(idx + i).rjust(length, '0')
            os.makedirs(os.path.join(output_folder, id), exist_ok=True)
            
            if save_obj:
                with open("{:s}/{:s}/joints.json".format(output_folder, id), 'w') as json_file:
                    json.dump(mano_joints[i].tolist(), json_file, indent=4)
                
                with open("{:s}/{:s}/manov.json".format(output_folder, id), 'w') as json_file:
                    json.dump(skin_mano_v[i].tolist(), json_file, indent=4)

                with open("{:s}/{:s}/pose_param.json".format(output_folder, id), 'w') as json_file:
                    json.dump(pose_param[i].tolist(), json_file, indent=4)
                
                with open("{:s}/{:s}/shape_param.json".format(output_folder, id), 'w') as json_file:
                    json.dump(shape_param[i].tolist(), json_file, indent=4)
                
                with open("{:s}/{:s}/tex_param.json".format(output_folder, id), 'w') as json_file:
                    json.dump(tex_param[i].tolist(), json_file, indent=4)
                
                with open("{:s}/{:s}/R.json".format(output_folder, id), 'w') as json_file:
                    json.dump(R[i].tolist(), json_file, indent=4)
            
            cur_path = os.path.join(output_folder, id)
            save_textured_nimble("{:s}/{:s}.obj".format(cur_path, id), skin_v_smooth[i], tex_img[i], save_obj=save_obj)
