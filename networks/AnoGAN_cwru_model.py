
import torch
import torch.nn as nn

# weight 초기화
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class Generator(nn.Module):

    def __init__(self, latent_dim=256, num_gf = 128, channels=1, bias=False):

        '''
        latent_dim: Latent vector dimension
        num_gf: Number of Generator Filters
        channels: Number of Generator output channels
        '''

        super(Generator, self).__init__()
        self.layer = nn.Sequential(

            # input size 1 --> 16
            nn.ConvTranspose1d(latent_dim, num_gf * 4, 16, 1, 0, bias=bias),
            nn.BatchNorm1d(num_gf * 4),
            nn.LeakyReLU(),
            # state size 16 --> 64
            nn.ConvTranspose1d(num_gf * 4, num_gf * 2, 6, 4, 1, bias=bias),
            nn.BatchNorm1d(num_gf * 2),
            nn.LeakyReLU(),
            # state size 64 --> 256
            nn.ConvTranspose1d(num_gf * 2, num_gf, 6, 4, 1, bias=bias),
            nn.BatchNorm1d(num_gf),
            nn.LeakyReLU(),
            # state size 256 --> 1024
            nn.ConvTranspose1d(num_gf, channels, 6, 4, 1, bias=bias),
            nn.Tanh()
        )

    def forward(self, z):

        z = self.layer(z)

        return z


class Discriminator(nn.Module):

    def __init__(self, num_df=128, channels=1, bias=False):
        super(Discriminator, self).__init__()
        self.feature_layer = nn.Sequential(

            # state size 1024 --> 256
            nn.Conv1d(channels, num_df, 6, 4, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),

            # state size 256 --> 64
            nn.Conv1d(num_df, num_df*2, 6, 4, 1, bias=bias),
            nn.BatchNorm1d(num_df*2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size 64 --> 16
            nn.Conv1d(num_df*2, num_df * 4, 6, 4, 1, bias=bias),
            nn.BatchNorm1d(num_df * 4),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.dis_layer = nn.Sequential(nn.Conv1d(num_df * 4, 1, 16, 1, 0, bias=bias),
                                       nn.Sigmoid())

    def forward_features(self, x):
        features = self.feature_layer(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        discrimination = self.dis_layer(features)
        return discrimination
