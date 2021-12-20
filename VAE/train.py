"""Training procedure for NICE.
"""

import argparse
import os
import torch, torchvision
from torch.utils.data import dataset
from torchvision import transforms
import numpy as np
# from VAE import model
# from VAE.VAE import Model
import matplotlib.pyplot as plt
import VAE

def train(vae, trainloader, optimizer, epoch, device):
    vae.train()  # set to training mode
    #TODO
    running_loss = 0
    batch_num = 1
    for inputs, _ in trainloader:
        batch_num += 1
        inputs = inputs.to(device)
        optimizer.zero_grad()
        reconstruction, mu,logvar = vae(inputs)
        loss = vae.loss(inputs, reconstruction,mu, logvar)
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    return running_loss / batch_num, mu, logvar


def test(vae, testloader, filename, epoch, device, sample_size):
    vae.eval()  # set to inference mode
    running_loss = 0
    with torch.no_grad():
        samples = vae.sample(sample_size).to(device)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),'./samples/' + filename + 'epoch%d.png' % epoch)
        for n_batches, data in enumerate(testloader, 1):
            inputs, _ = data
            inputs = inputs.to(device)
            x_recon, mu, logvar = vae(inputs)
            loss = vae.loss(inputs, x_recon, mu=mu, logvar=logvar)
            running_loss += float(loss)
    return running_loss / n_batches

def init_loaders(dataset, batch_size, transform):
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=batch_size, shuffle=False, num_workers=2)
    elif dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')
    return trainloader,testloader



def run_train_test(epochs, device, trainloader, testloader, filename, vae, optimizer, sample_size):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss, mu, logvar = train(vae=vae, trainloader=trainloader, optimizer=optimizer,
                                       epoch=epoch, device=device)
        train_losses.append(train_loss)
        val_loss = test(vae=vae, testloader=testloader,filename=filename,epoch=epoch, device=device, sample_size=sample_size)
        val_losses.append(val_loss)
    return train_losses,val_losses



def plot_losses(dataset, train_losses, val_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(val_losses)
    ax.set_title("Train/Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["train loss","test loss"])
    plt.savefig(os.path.join(os.getcwd(),"loss",f"{dataset}_loss.png"))
        

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    trainloader, testloader = init_loaders(args.dataset, args.batch_size, transform)

    filename = '%s_' % args.dataset \
             + 'batch_size%d_' % args.batch_size \
             + 'latent_dim%d_' % args.latent_dim

    vae = VAE.Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    train_losses, val_losses = run_train_test(args.epochs, device, trainloader, testloader,\
         filename, vae, optimizer,sample_size=args.sample_size)

    plot_losses(args.dataset, train_losses, val_losses)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
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
