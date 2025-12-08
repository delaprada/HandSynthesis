from torch.autograd import Variable
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, z_dim, encoder, decoder):
        """
        the VAE model,
        :param z_dim: int
        :param encoder: nn.Model
        :param decoder: nn.Model
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        h_i = self.encoder(x)
        return h_i[:, self.z_dim:], h_i[:, :self.z_dim]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            mu

    def forward(self, x, vae_decoder=None, hand_size=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec = vae_decoder if vae_decoder else self.decoder
        return dec(z), mu, logvar
