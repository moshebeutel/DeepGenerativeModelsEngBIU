"""Training procedure for NICE.
"""

import argparse
import os
import torch, torchvision
from torchvision import transforms
import numpy as np
import GAN
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
def train(G, D, trainloader, optimizer_G, optimizer_D, epochs, device, sample_size,
          dataset, loss_type, visualize_every=10):
    G.train()  # set to training mode
    D.train()
    #TODO
    D_losses, G_losses = [], []
    D_losses_vs_epochs, G_losses_vs_epochs = [], []
    criterion = nn.BCELoss()
    epoch_pbar =  tqdm(range(1,epochs+1))
    for epoch in epoch_pbar:
        for (inputs, _) in trainloader:
            fake_batch = generate_fakes(G, trainloader, device)
            train_discriminator(D, trainloader, optimizer_D, device, D_losses, criterion, inputs, fake_batch)
            train_generator(G, D, trainloader, optimizer_G, device, loss_type, G_losses, criterion, fake_batch)
            

        discriminator_loss = torch.mean(torch.FloatTensor(G_losses))
        generator_loss = torch.mean(torch.FloatTensor(D_losses))
        G_losses_vs_epochs.append(discriminator_loss)
        D_losses_vs_epochs.append(generator_loss)
        epoch_pbar.set_postfix({'epoch': epoch, 'Discriminator Loss': discriminator_loss.item(), 'Generator Loss': generator_loss.item()})
        if (epoch+1) % visualize_every == 0: 
            sample(G, sample_size, device, save_file=True, epoch=epoch+1, dataset=dataset, loss_type=loss_type)

    return G_losses_vs_epochs, D_losses_vs_epochs

def generate_fakes(G, trainloader, device):
    z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
    fake_batch = G(z)
    return fake_batch

def train_discriminator(D, trainloader, optimizer_D, device, D_losses, criterion, inputs,fake_batch, num_discrimanator_trains_per_generator_train = 1):
    
    for _ in range(num_discrimanator_trains_per_generator_train): 
        D.zero_grad()
        D_real_loss = train_discrimanator_on_real(D, device, criterion, inputs)
        D_fake_loss = train_discriminator_on_fake(D, trainloader, device, criterion, fake_batch)

        # backward was done for both losses - accumulate gradients
        D_loss = D_real_loss + D_fake_loss

        optimizer_D.step()

    D_losses.append(D_loss.item())

def train_discriminator_on_fake(D, trainloader, device, criterion, fake_batch):
    y_fake = torch.zeros(trainloader.batch_size, 1).to(device)
    D_out = D(fake_batch.detach())
    D_fake_loss = criterion(D_out, y_fake)
    D_fake_loss.backward()
    return D_fake_loss

def train_discrimanator_on_real(D, device, criterion, inputs):
    x_real = inputs.view(-1, 784).to(device)
    y_real = torch.ones(x_real.size(0), 1).to(device)  # labels of real inputs is 1
    D_out = D(x_real)
    D_real_loss = criterion(D_out, y_real)
    D_real_loss.backward()
    return D_real_loss

def train_generator(G, D, trainloader, optimizer_G, device, loss_type, G_losses, criterion, fake_batch):
    G.zero_grad()
    assert loss_type in ['original', 'standard']
    # BCE Loss -  -[y_n log(x_n) + (1 - y_n) log(1 - x_n))]
    # so ones label vector give: -log(x_n) 
    #    zeros label vector give: -log(1-x_n)
    # standard - maximizing E[log(D(G(z)))] -  ones vector - minimize (-log(D(G(z))))
    # original - minimize E[log(1-D(G(z)))] -  zeros vector- minimize -(-log(1-D(G(z))))
    y = torch.ones(trainloader.batch_size, 1).to(device) if loss_type == 'standard' \
         else torch.zeros(trainloader.batch_size, 1).to(device)  # zeros instead of ones

    DGz = D(fake_batch)
    Generator_loss = criterion(DGz, y)
    
    #backprop
    if loss_type == 'original': # 'original' -  minimize E[log(1-D(G(z)))]
        neg_G_loss = torch.neg(Generator_loss)
        neg_G_loss.backward()  # Minimizing the negative loss is as maximizing the loss
    else:                       # 'standard' - maximizing E[log(D(G(z)))]
        assert loss_type == 'standard'
        Generator_loss.backward()

    optimizer_G.step()

    G_losses.append(Generator_loss.item())





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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # trainloader returns an error when this normalization is used
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])

 

    
    for dataset in ['mnist', 'fashion-mnist']:
        root = f'./data/{dataset}'
        trainset = torchvision.datasets.FashionMNIST(root= root,train=True, download=True, transform=transform) if dataset == 'fashion-mnist' else\
            torchvision.datasets.MNIST(root=root,train=True, download=True, transform=transform)
        trainloader= torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True, num_workers=2)
        for loss_type in ['original', 'standard']:
            G = GAN.Generator(latent_dim=args.latent_dim,batch_size=args.batch_size, device=device).to(device)
            D = GAN.Discriminator().to(device)
            optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
            #TODO
            G_losses, D_losses = train(G,D, trainloader, optimizer_G, optimizer_D, args.epochs,\
                device,  args.sample_size, dataset=dataset, loss_type=loss_type, visualize_every=10)

            plot_learning_curves(dataset, loss_type, G_losses, D_losses)

def plot_learning_curves(dataset, loss_type, G_losses, D_losses):
    plt.figure()
    plt.plot(G_losses)
    plt.plot(D_losses)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Loss vs Epoch')
    plt.savefig(os.path.join(os.getcwd(),"loss",f'{dataset}_loss_type_{loss_type}.png'))
    plt.cla()

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
                        help='either maximize generator loss with \'standard\' or minimize with \'original\'',
                        type=str,
                        default='original')
                        # default='standard')

    args = parser.parse_args()
    main(args)
