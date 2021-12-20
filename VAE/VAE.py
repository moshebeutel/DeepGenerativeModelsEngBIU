"""NICE model
"""

from math import log
from numpy.lib.function_base import select
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.encoder_output_shape = (64,7,7)
        self.encoder_output_size = 64*7*7

        self.mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.logvar = nn.Linear(self.encoder_output_size, latent_dim)

        self.upsample = nn.Linear(latent_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self,sample_size):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        with torch.no_grad():
            z = torch.rand((sample_size, self.latent_dim)).to(self.device)
            return self.decoder(self.upsample(z).view(-1, *self.encoder_output_shape))
        
    
    def z_sample(self, mu, logvar):
        '''
        :param mu: model mean
        :param logvar: model log of variance - log(sigma^2)
        :return float: sample from the latent space
        '''
        # Use Reparametrization trick - 
        #   epsilon ~ U(0,1)
        #   g(epsilon) = mu + sigma * epsilon
        #   z = g(epsilon, (mu, sigma))
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.rand_like(sigma).to(self.device)
        return mu + sigma * epsilon

    def loss(self,x,x_reconstruction,mu,logvar):
        def KL(mu, logvar):
            # Use KL of two gaussians KL(N(mu, exp(logvar)) || N(0,1))
            return torch.mul(torch.sum(-1-logvar+torch.pow(mu,2.0)+torch.exp(logvar)),0.5)
        def binary_cross_entropy(x, x_reconstruction):
            return F.binary_cross_entropy(x_reconstruction, x, reduction='sum')
        batch_size = x.shape[0]
        return (binary_cross_entropy(x,x_reconstruction) + KL(mu,logvar)) / batch_size

    def forward(self, x):
        latent = self.encoder(x).view(-1,self.encoder_output_size)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        z = self.z_sample(mu, logvar)
        up_sampled = self.upsample(z).view(-1,*self.encoder_output_shape)
        x_reconstruction = self.decoder(up_sampled)
        return x_reconstruction, mu, logvar