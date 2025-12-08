import torch
import json
import copy
import numpy as np

class HO3DDataset(torch.utils.data.Dataset):
  def __init__(self, datalist, gt_datalist, occ_gt_datalist):
    self.datalist = datalist
    self.gt_datalist = gt_datalist
    self.occ_gt_datalist = occ_gt_datalist
  
  def __len__(self):
    return len(self.datalist)

  def __getitem__(self, idx):
    data = copy.deepcopy(self.datalist[idx])
    gt_data = copy.deepcopy(self.gt_datalist[idx])
    occ_info = copy.deepcopy(self.occ_gt_datalist[idx])
    occ_mask = np.array(occ_info, dtype=bool)
    mask = ~occ_mask
    
    input = {
      "d3d": data,
      "gt_d3d": gt_data,
      "mask": mask
    }
    
    return input
  