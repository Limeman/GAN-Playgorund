import torch.nn as nn

class Discriminator(nn.Module):
    """
        The model discriminator class
    """
    def __init__(self, nc, ndf, ngpu):
        """
            Initializes the discriminator model

            ndf: int
                The size of the feature maps of the discriminator
            nc: int
                The number of color channels in the images
            ngpu: int
                The number of gpu's to use in training
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is an image, real or fake that is nc * 64 * 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size is ndf * 32 * 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size is (ndf * 2) * 16 * 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size is (ndf * 4) * 8 * 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size is (ndf * 8) * 4 * 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        """
                The forward pass of the discriminator. Takes in an
            image and decides the probability that it is true or false
        """
        return self.main(input)