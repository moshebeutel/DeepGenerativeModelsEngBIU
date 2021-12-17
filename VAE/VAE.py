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

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
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
            # z = self.z_sample(mu, logvar)
            z = torch.rand((sample_size, self.latent_dim)).to(self.device)
            return self.decoder(self.upsample(z).view(-1, 64, 7, 7))
        
    
    def z_sample(self, mu, logvar):
        '''
        :param mu: 
        :param logvar: 
        :return 
        '''
        #TODO
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std).to(self.device)
        return mu + eps * std

    def loss(self,x,recon,mu,logvar):
        #TODO
        def f(z):
            return 1/(2*torch.pi * logvar) * torch.exp(((z-mu)**2) / logvar**2)
        def KL(mu, logvar):
            return torch.mul(torch.sum(1+ logvar-torch.pow(mu,2.0)-torch.exp(logvar)),-0.5)
        def binary_cross_entropy(x, recon):
            # return torch.dot(x, torch.log(f_z)) + torch.dot(1-x,torch.log(1-f_z))
            return F.binary_cross_entropy(recon, x, reduction='sum')
        bce = binary_cross_entropy(x,recon)
        return binary_cross_entropy(x,recon) + KL(mu,logvar)

    def forward(self, x):
        #TODO
        latent = self.encoder(x).view(-1,64*7*7)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        z = self.z_sample(mu, logvar)
        up_sampled = self.upsample(z).view(-1,64,7,7)
        reconstructed = self.decoder(up_sampled)
        return reconstructed, mu, logvar