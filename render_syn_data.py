import bpy
import bpycv
import sys
import math
import os
import json
import cv2
import random
import numpy as np

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

import utils
from utils.default_config import obj_trans, camera_settings, tex_wall_settings
from utils.components import \
    smooth_obj, \
    set_output_resolution, \
    create_background, \
    add_light, \
    add_named_material, \
    add_hdri_background, \
    remove_nodes, \
    remove_materials
from utils.camera import get_calibration_matrix_K_from_blender, get_4x4_RT_matrix_from_blender

resolution_percentage = int(sys.argv[sys.argv.index('--') + 1])
num_samples = int(sys.argv[sys.argv.index('--') + 2])
start_count = int(sys.argv[sys.argv.index('--') + 3])

# load configuration from JSON file
with open('./config_syn_data.json') as config_file:
    config = json.load(config_file)

# Reset
utils.clean_objects()

folder_path = config["data_path"]
# mesh_path = config["mesh_path"]
samples = sorted(os.listdir(folder_path))
# meshes = sorted(os.listdir(mesh_path))
hdri_files = sorted(os.listdir(config["hdri_bg_path"]))
cam_RT_list = []

start = start_count
count = start + 1 # for better parallel execution on multiple processes

for i in range(start, count):
    output_path = os.path.join(config["output_path"], 'rgb' + '_' + str(i+1))
    output_seg_path = os.path.join(config["output_path"], 'segmentation' + '_' + str(i+1))
    output_depth_path = os.path.join(config["output_path"], 'depth' + '_' + str(i+1))
    
    if not os.path.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    if config["if_seg"] and not os.path.exists(output_seg_path):
        os.mkdir(output_seg_path)
    
    if config["if_depth"] and not os.path.exists(output_depth_path):
        os.mkdir(output_depth_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for idx, sample in enumerate(samples):
        utils.clean_objects()
        
        sample = samples[idx]
        # mesh = meshes[idx]

        # multi-view rendering
        if config["multi_view"]:
            viewpoint = random.randint(0, len(camera_settings) - 1)
        else:
            viewpoint = 0
        
        # scene building
        # add light
        if config["add_light"]:
            location = tuple(config["light_location"])
            
            rot_x, rot_y, rot_z = config["light_rot"]
            rotation = (math.radians(rot_x),
                        math.radians(rot_y), math.radians(rot_z))

            scale = tuple(config["light_scale"])
            energy = config["light_strength"]
            add_light(config["light_type"], location=location,
                    rotation=rotation, scale=scale, energy=energy)

        if config["add_hand_tex"]:
            skin_mesh_path = os.path.join(folder_path, sample, 'skin.obj')
            diffuse_map_path = os.path.join(folder_path, sample, 'diffuse.png')
            specular_map_path = os.path.join(folder_path, sample, 'spec.png')
            normal_map_path = os.path.join(folder_path, sample, 'normal.png')

            texture = {
                "ambient_occlusion": "",
                "color": os.path.join(working_dir_path, diffuse_map_path),
                "displacement": "",
                "metallic": "",
                "normal": os.path.join(working_dir_path, normal_map_path),
                "roughness": "",
                "specular": os.path.join(working_dir_path, specular_map_path),
            }

            add_named_material(sample, texture)

        # import hand mesh to the scene
        bpy.ops.import_scene.obj(filepath=skin_mesh_path)
        skin_obj = bpy.context.selected_objects[0]

        # transform obj to camera coordinate system
        skin_obj.location = obj_trans['location']
        skin_obj.rotation_mode = obj_trans['rotation_mode']
        skin_obj.rotation_euler = obj_trans['rotation_euler']
        skin_obj.scale = obj_trans['scale']

        # assign material to object
        if skin_obj.data.materials:
            # assign to 1st material slot
            skin_obj.data.materials[0] = bpy.data.materials[str(sample)]
        else:
            # no slots
            skin_obj.data.materials.append(bpy.data.materials[str(sample)])

        # smooth mesh
        smooth_obj(bpy.ops.object)
            
        # add camera and set camera location
        cam = camera_settings[viewpoint]
        bpy.ops.object.camera_add(
            location=cam['location'], rotation=cam['rotation'])

        camera_obj = bpy.context.object
        camera_obj.rotation_mode = 'XYZ'
        
        # Camera sensor settings
        # f/rou = K[0]; f = K[0] * rou; unit of rou is meter/pixel => 36 mm / 224 px (since sensor width is 36 mm)
        # unit of lens is also mm, so no need to transform to meter
        sensor_width = 36
        res_w = config["res_w"]
        lens = 355.0 * sensor_width / res_w # set physical focal length
        utils.set_camera_params_nimble(camera_obj.data, skin_obj, lens=lens)
        
        # render setting
        scene = bpy.data.scenes["Scene"]
        set_output_resolution(scene, resolution_percentage,
                            res_x=config["res_h"], res_y=config["res_w"])

        # add RGB background
        if config["add_rgb_bg"]:
            bg_rgb = config["bg_color"]
            bg_strength = config["bg_strength"]
            create_background(scene=scene, rgb=bg_rgb, strength=bg_strength)

        # add HDRI background
        if config["add_hdri_bg"]:
            # choose random background from HDRI assets
            idx = random.randint(0, int(len(hdri_files) * config["hdri_ratio"] - 1))
            
            hdri_path = os.path.join(
                working_dir_path, config["hdri_bg_path"], hdri_files[idx])
            
            # add rotation on background to increase diversity
            rotation = random.uniform(-180, 180)
            add_hdri_background(scene=scene, hdri_path=hdri_path, rotation=rotation)
        
        if config["add_tex_bg"]:
            tex_bg_path = config["tex_bg_path"]
            bg_texture = {
                "color": os.path.join(working_dir_path, tex_bg_path, "col.jpg"),
                "displacement": os.path.join(working_dir_path, tex_bg_path, "disp.jpg"),
                "normal": os.path.join(working_dir_path, tex_bg_path, "nrm.jpg"),
                "roughness": os.path.join(working_dir_path, tex_bg_path, "rgh.jpg"),
                "ambient_occlusion": "",
                "metallic": "",
                "specular": "",
            }

            # add material
            add_named_material("Marble01", bg_texture, displacement_scale=0.2)
            
            # get texture wall setting
            tex_wall = tex_wall_settings[viewpoint]

            # create a plane
            current_object = utils.create_plane(size=tex_wall["size"],
                                                location=tex_wall["location"],
                                                rotation=tex_wall["rotation"],
                                                name="Wall")
            # add texture to the plane
            current_object.data.materials.append(bpy.data.materials["Marble01"])
        
        utils.set_cycles_renderer(scene, camera_obj, num_samples, use_transparent_bg=False)
        
        # use bpycv to render rgb, depth map, segmentation
        obj = bpy.context.active_object
        result = bpycv.render_data()

        # transfer RGB image to opencv's BGR
        cv2.imwrite(os.path.join(output_path, sample + '.png'), result["image"][..., ::-1])

        # render segmentation mask and depth map
        # convert depth units from meters to millimeters
        depth_in_mm = result["depth"] * 1000
        
        if config["if_seg"]:
            # save as 16 bit png
            cv2.imwrite(os.path.join(output_seg_path, sample + '.png'), depth_in_mm)
        
        if config["if_depth"]:
            res_w = config["res_w"]
            depth_map = cv2.cvtColor(result.vis()[:res_w, res_w * 2:, :3], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_depth_path, sample + '.png'), depth_map)
        
        # delete hdri nodes and materials to avoid slowing down render speed
        remove_nodes(scene)
        remove_materials()

        # obtain the object R, T during rendering process
        for obj in scene.objects:
            if obj.type == 'CAMERA':
                K, w, h= get_calibration_matrix_K_from_blender(obj.data)
                R, T, RT = get_4x4_RT_matrix_from_blender(obj)
                            
                cam_RT = np.eye(4)
                cam_RT[:3, :3] = np.array(RT)[:3, :3]
                cam_RT[:3, 3] = np.array(RT)[:3, 3]
                
                np.set_printoptions(precision=4, suppress=True)
                cam_RT_list.append(cam_RT.tolist()) # cam_RT
        
        # delete unused data block to avoid slowing down render speed
        bpy.ops.outliner.orphans_purge()

with open(os.path.join(config["output_path"][:-6], "cam_RT_syn_data_" + str(start_count) + ".json"), 'w') as json_file:
    json.dump(cam_RT_list, json_file, indent=4)
