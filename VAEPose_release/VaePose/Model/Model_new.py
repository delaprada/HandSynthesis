from torch.autograd import Variable
import torch.nn as nn
import torch

# basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class JointEncoder(nn.Module):

    def __init__(self, in_dim, z_dim):
        super(JointEncoder, self).__init__()
        in_dim = torch.IntTensor(in_dim)
        self.in_size = in_dim.prod()

        self.inp_post = nn.Linear(self.in_size, 1024)
        enc_post = []
        for i in range(2):
            enc_post.append(ResidualBlock(1024))
        self.enc_post = nn.Sequential(*enc_post)
        self.out_post = nn.Linear(1024, z_dim*2)
        # additional layers
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(z_dim*2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # return self.linear_layers(x.view(-1, self.in_size))
        # post_inp = torch.cat([x, y], dim=1)
        post_inp = self.inp_post(x.view(-1, self.in_size))
        post_inp = self.batch_norm1(post_inp)
        post_inp = self.relu(post_inp)
        post_inp = self.dropout(post_inp)

        post = self.enc_post(post_inp)
        post = self.out_post(post)
        post = self.batch_norm2(post)
        post = self.relu(post)
        post = self.dropout(post)
        
        return post


class JointDecoder(nn.Module):

    def __init__(self, z_dim, out_dim):
        super(JointDecoder, self).__init__()
        self.out_dim = torch.IntTensor(out_dim)
        self.out_size = self.out_dim.prod()
        self.fc_in = nn.Linear(z_dim, 1024)
        dec = []
        for i in range(2):
            dec.append(ResidualBlock(1024))
        self.dec = nn.Sequential(*dec)
        self.fc_out = nn.Linear(1024, self.out_size.data.numpy())

    def forward(self, x):
        
        out = self.fc_in(x)
        out = self.dec(out)
        out = self.fc_out(out)
        
        return out.view(-1, self.out_dim[0], self.out_dim[1])
