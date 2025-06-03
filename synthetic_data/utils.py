import os
import json
import torch
import numpy as np

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2), value=1)
    return out_uv

def intrinsic_matrix(fx, fy, cx, cy, skew=0):
    K = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K
