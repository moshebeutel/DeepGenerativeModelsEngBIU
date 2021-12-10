"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
# from VAE import model
from VAE.VAE import Model

def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    #TODO
    running_loss = 0
    batch_num = 1
    for inputs, _ in trainloader:
        batch_num += 1
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # change  shape from BxCxHxW to Bx(C*H*W)
        # inputs = inputs.to(device)
    
        optimizer.zero_grad()
        loss = -vae(inputs).mean()
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    return running_loss / batch_num


def test(vae, testloader, filename, epoch):
    vae.eval()  # set to inference mode
    running_loss = 0
    with torch.no_grad():
        #TODO    
        num_samples = 100
        samples = vae.sample(num_samples).gpu()
        samples = samples.view(num_samples, samples.shape[0] , samples.shape[1] , samples.shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                        './samples/' + filename + 'epoch%d.png' % epoch)

        for n_batches, data in enumerate(testloader, 1):
            inputs, _ = data
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # change  shape from BxCxHxW to Bx(C*H*W)
            # inputs = inputs.to(device)
            loss = -vae(inputs).mean()
            running_loss += float(loss)
    return running_loss / n_batches

        

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    #TODO

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
