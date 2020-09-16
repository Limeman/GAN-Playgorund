import torch.nn as nn

class Generator(nn.Module):
    """
        The model generator class
    """
    def __init__(self, nz, ngf, nc, ngpu):
        """
            Initializes the generator model

            nz: int
                The size of the latent space z
            ngf: int
                The size of the feature maps of the generator
            nc: int
                The number of color channels in the resulting images
            ngpu: int
                The number of gpu's to use in training
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size (ngf * 4) * 8 * 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size (ngf * 2) * 16 * 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size ngf * 32 * 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        """
                The forward pass of the generator, takes in a latent space input z
            and returns a generated image.
        """
        return self.main(input)