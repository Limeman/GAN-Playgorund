# Dependencies

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Custom module imports
from _utils._utils import weights_init
from Models.generator import Generator
from Models.discriminator import Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--seed', required=False, help='An integer for seeding random number generation', default=1337)
parser.add_argument('--dataroot', required=False, help='The root path to the dataset', default='../Data/celeba')
parser.add_argument('--workers', required=False, help='The number of active workers in the dataloader', default=0)
parser.add_argument('--batch_size', required=False, help='The batch size used when training', default=128)
parser.add_argument('--image_size', required=False, help='The pixel dimensionality of the generated and real images', default=64)
parser.add_argument('--nc', required=False, help='The number of color channels in each image', default=3)
parser.add_argument('--nz', required=False, help='The dimensionality of the latent space z', default=100)
parser.add_argument('--ngf', required=False, help='The size of the feature maps in the generator', default=64)
parser.add_argument('--ndf', required=False, help='The size of the feature maps in the discriminator', default=64)
parser.add_argument('--num_epochs', required=False, help='The number of epochs of training', default=5)
parser.add_argument('--lr', required=False, help='The learning rate for the optimizers', default=0.0002)
parser.add_argument('--beta1', required=False, help='The beta1 hyperparameter for Adam optimizer', default=0.5)
parser.add_argument('--ngpu', required=False, help='The number of gpus in the system', default=1)


opt = parser.parse_args()

# Set the seed for RNG
random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Create the dataset and dataloader
dataset = dset.ImageFolder(root=opt.dataroot, transform=transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

# Determine the device(s) that pytorch can run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

# Create the generator
netG = Generator(opt.nz, opt.ngf, opt.nc, opt.ngpu).to(device)

# Handle multi gpu
if (device.type == 'cuda' and (opt.ngpu > 1)):
    netG = nn.DataParallel(netG, list(range(opt.ngpu)))

# Initialize the model weights
netG.apply(weights_init)

# Create the discriminator
netD = Discriminator(opt.nc, opt.ndf, opt.ngpu).to(device)

# Handle multi gpu
if (device.type == 'cuda' and (opt.ngpu > 1)):
    netD = nn.DataParallel(netD, list(range(opt.ngpu)))

netD.apply(weights_init)

criterion = nn.BCELoss()

# Create a batch of fized latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

# Define the real and fake label convention
real_label = 1
fake_label = 0

# Setup an optimizer for each model
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# |                                     Model Training                                            |
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting up the training')

for epoch in range(opt.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ##################
        # Step 1, update the discriminator model by maximizing: log(D(x)) + log(1 - D(G(z)))
        ##################
        ## First, train with an all-real batch
        netD.zero_grad()
        # Format the batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
        # Forward pass the real batch through the discriminator
        output = netD(real_cpu).view(-1)
        # Calculate the loss for the batch of real data
        errD_real = criterion(output, label)
        # Get the gradient for D
        errD_real.backward()
        D_x = output.mean().item()

        ## Then, train with an all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
        # Generate fake images using the generator
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify the fake batch using the discriminator
        output = netD(fake.detach()).view(-1)
        # Calculate the discriminators loss on the fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradient for the fake batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Finally, update the discriminator
        optimizerD.step()
        
        ##################
        # Step 2, update the generator model by maximizing: log(D(G(z)))
        ##################
        netG.zero_grad()
        label.fill_(real_label) # Fake labels are real for generator cost
        # Since D was just updated, perform another forward pass of all-fake
        # batch through D
        output = netD(fake).view(-1)
        # Calculate the generators loss
        errG = criterion(output, label)
        # Calculate the gradient for the generator
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update the generators parameters
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # Save the loss for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# Save the models
torch.save(netD.state_dict(), 'Models/discriminator.pt')
torch.save(netG.state_dict(), 'Models/generator.pt')

# Save the stats
torch.save({'G_losses' : G_losses, 'D_losses' : D_losses}, 'Models/training_stats.pt')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

