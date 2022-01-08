"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import GAN
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import save_image


# N = 100

def train(G, D, trainloader, optimizer_G, optimizer_D, epoch, device, sample_size,
          dataset, loss_type, visualize_every=10):
    G.train()  # set to training mode
    D.train()
    #TODO
    D_losses, G_losses = [], []
    D_losses_vs_epochs, G_losses_vs_epochs = [], []
    criterion = nn.BCELoss()

    for i in range(1, epoch + 1):
        for batch_idx, (inputs, _) in enumerate(trainloader):

            G.zero_grad()

            if loss_type == 'modified':
                z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
                y = torch.ones(trainloader.batch_size, 1).to(device)
                G_out = G(z)
                D_out = D(G_out)
                G_loss = criterion(D_out, y)
                G_loss.backward()

            elif loss_type == 'standard':
                z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
                y = torch.zeros(trainloader.batch_size, 1).to(device)  # zeros instead of ones
                G_out = G(z)
                D_out = D(G_out)
                G_loss = criterion(D_out, y)
                neg_G_loss = -G_loss
                (neg_G_loss).backward()  # Minimizing the negative loss is as maximizing the loss
            else:
                raise ValueError('loss_type should be either \'standard\' or \'modified\' ')

            # gradient backprop & optimize ONLY G's parameters
            optimizer_G.step()
            G_losses.append(G_loss.item())

            for k in range(2):  # Training the discriminator multiple times per each generator iteration
                D.zero_grad()

                # train discriminator on real samples
                x_real = inputs.view(-1, 784).to(device)
                y_real = torch.ones(x_real.size(0), 1).to(device)  # labels of real inputs is 1

                D_out = D(x_real)
                D_real_loss = criterion(D_out, y_real)

                # train discriminator on fake samples
                z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
                x_fake = G(z)
                y_fake = torch.zeros(trainloader.batch_size, 1).to(device)

                D_out = D(x_fake)
                D_fake_loss = criterion(D_out, y_fake)

                # Backprop
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizer_D.step()

            D_losses.append(D_loss.item())

        print('Epoch[%d/%d]: Loss Disc: %.3f, Loss Gen: %.3f'
              % ((i), epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
        G_losses_vs_epochs.append(torch.mean(torch.FloatTensor(G_losses)))
        D_losses_vs_epochs.append(torch.mean(torch.FloatTensor(D_losses)))

        if (i+1) % visualize_every == 0: # Visualize every 10 epochs
            sample(G, sample_size, device, save_file=True, epoch=i+1, dataset=dataset, loss_type=loss_type)

    return G_losses_vs_epochs, D_losses_vs_epochs





def sample(G, sample_size, device, save_file, epoch, dataset, loss_type):
    G.eval()  # set to inference mode
    with torch.no_grad():
        #TODO
        test_z = torch.randn(sample_size, G.latent_dim).to(device)
        generated = G(test_z)
        # Shift it back to the range [0,1]:
        generated = (generated + 1)/2
        if save_file:
            save_image(generated.view(generated.size(0), 1, 28, 28),
                       './samples/sample_' + loss_type + '_loss_' + dataset +'_epoch' + str(epoch) + '.png')



def main(args):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # trainloader returns an error when this normalization is used
        transforms.Normalize(mean=(0.5), std=(0.5)),
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

    G = GAN.Generator(latent_dim=args.latent_dim,
                      batch_size=args.batch_size, device=device).to(device)
    D = GAN.Discriminator().to(device)

    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(
        D.parameters(), lr=args.lr)

    #TODO

    G_losses, D_losses = train(G,D, trainloader, optimizer_G, optimizer_D, args.epochs,
                               device,  args.sample_size, dataset=args.dataset, loss_type=args.loss_type, visualize_every=2)
    G_losses, D_losses = train(G,D, trainloader, optimizer_G, optimizer_D, 30,
                               device,  args.sample_size, dataset=args.dataset, loss_type='standard', visualize_every=10)

    # Plot Generator and Discriminator learning curves:
    plt.figure()
    plt.plot(G_losses)
    plt.plot(D_losses)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Loss vs Epoch')


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
                        default=100)
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
                        default=2e-4)
    parser.add_argument('--loss_type',
                        help='either maximize generator loss with \'standard\' or minimize with \'modified\'',
                        type=str,
                        default='modified')
                        # default='standard')

    args = parser.parse_args()
    main(args)
