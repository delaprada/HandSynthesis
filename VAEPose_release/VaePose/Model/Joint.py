import torch.nn as nn
import torch


class JointEncoder(nn.Module):

    def __init__(self, in_dim, z_dim):
        super(JointEncoder, self).__init__()
        in_dim = torch.IntTensor(in_dim)
        self.in_size = in_dim.prod()
        self.linear_layers = nn.Sequential(
            nn.Linear(self.in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # add 2 more linear layer
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),

            nn.Linear(512, 2 * z_dim))

    def forward(self, x):
        return self.linear_layers(x.view(-1, self.in_size))


class JointDecoder(nn.Module):

    def __init__(self, z_dim, out_dim):
        super(JointDecoder, self).__init__()
        self.out_dim = torch.IntTensor(out_dim)
        self.out_size = self.out_dim.prod()
        self.linear_layers = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_size.data.numpy()))

    def forward(self, x):
        out = self.linear_layers(x)
        return out.view(-1, self.out_dim[0], self.out_dim[1])
