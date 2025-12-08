import torch.nn as nn
from . import RGBEncoder

class VideoEncoder(nn.Module):
    def __init__(self, n_frames, z_dim, hand_side_invariance):
        super(VideoEncoder, self).__init__()
        self.n_frames = n_frames
        self.z_dim = z_dim
        self.hand_side_invariance = hand_side_invariance
        self.frameEncoder = RGBEncoder(z_dim, hand_side_invariance)
        features_num = self.n_frames * self.z_dim * 2
        self.layers = nn.Sequential(
            nn.Linear(features_num, features_num),
            nn.ReLU(),
            # nn.BatchNorm1d(features_num),
            nn.Linear(features_num, features_num),
            nn.ReLU(),
            # nn.BatchNorm1d(features_num),
            nn.Linear(features_num, features_num))

    def forward(self, x):
        assert x.size(0) == self.n_frames
        y = self.frameEncoder(x)
        y = y.view(1, -1)
        res = y
        y = self.layers(y)
        o = res + y
        o = o.view(self.n_frames, -1)
        return o