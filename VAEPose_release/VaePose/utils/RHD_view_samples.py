""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

RHD_ROOT = " E:\\Data\\RHD\\RHD\\"

# chose between training and evaluation set
# set = 'training'
set = 'evaluation'
data_path = os.path.join(RHD_ROOT, set)

data_path = 'E:\\Data\\RHD\\RHD\\evaluation\\'

# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

# load annotations of this set
with open(os.path.join(data_path, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# iterate samples of the set
for sample_id, anno in anno_all.items():
    # load data
    image = scipy.misc.imread(os.path.join(data_path, 'color', '%.5d.png' % sample_id))
    mask = scipy.misc.imread(os.path.join(data_path, 'mask', '%.5d.png' % sample_id))
    depth = scipy.misc.imread(os.path.join(data_path, 'depth', '%.5d.png' % sample_id))

    # process rgb coded depth into float: top bits are stored in red, bottom in green channel
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # depth in meters from the camera

    # get info from annotation dictionary
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
    kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
    camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

    # Project world coordinates into the camera frame
    kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

    # Visualize data
    fig = plt.figure(1)
    ax1 = fig.add_subplot('221')
    ax2 = fig.add_subplot('222')
    ax3 = fig.add_subplot('223')
    ax4 = fig.add_subplot('224', projection='3d')

    ax1.imshow(image)
    ax1.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
    ax2.imshow(depth)
    ax3.imshow(mask)
    ax4.scatter(kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')

    plt.show()

    break
