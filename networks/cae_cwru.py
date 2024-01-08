

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
# from torch import optim
import sys
import matplotlib.pyplot as plt
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
from Bar import Bar


class Convolutional_1D_autoencoder(nn.Module):

    def __init__(self, z_dim = 16):
        super(Convolutional_1D_autoencoder, self).__init__()

        self.z_dim = z_dim
        self.pool = nn.MaxPool1d(4)
        # Conv2d(input channel, output channel, kernel_size) --> bias can affect model collapse problem (고려 X)
        self.conv1 = nn.Conv1d(1, 8, 5, bias=False, padding=2)
        # BatchNorm2d(num_features, eps = 1e-05, affine)
        self.bn1 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        # Linear(in_features, out_features, bias = True)
        self.fc1 = nn.Linear(16 * z_dim, z_dim, bias=False)

        # ConvTraspose2d(in_channels, out_channels, kernel_size)

        self.bn3 = nn.BatchNorm1d(2, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose1d(2, 4, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose1d(4, 8, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose1d(8, 1, 5, bias=False, padding=2)

    # init 함수에서 정의한 레이어중 encode 부분에서 사용할 layer를

    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(x, negative_slope= 0.2))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(x, negative_slope= 0.2))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, x):

        x = x.view(x.size(0), int(self.z_dim / 8), 8)
        # the input tensor, scale_factor : multiplier for spatial size
        x = F.interpolate(F.leaky_relu(x, negative_slope=0.2), scale_factor=8)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(x, negative_slope=0.2), scale_factor=4)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(x, negative_slope=0.2), scale_factor=4)
        x = self.deconv3(x)

        return torch.sigmoid(x)

    def forward(self, x):

        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat

def train(args, train_loader, device):

    """Training the Deep SVDD model"""
    net = Convolutional_1D_autoencoder().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_milestones, gamma=0.1)

    net.train()
    loss_values = []
    z_ = []

    for epoch in range(args.num_epochs):

        total_loss = 0

        for x, _ in Bar(train_loader):

            x = x.float().to(device)
            optimizer.zero_grad()
            z = net.encode(x)
            x_hat = net.decode(z)
            reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
            reconst_loss.backward()
            optimizer.step()

            total_loss += reconst_loss.item()

            if epoch == (args.num_epochs-1):

                z_.append(z.cpu().detach())

        scheduler.step()
        print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss / len(train_loader)))
        loss_values.append(total_loss/ len(train_loader))

    plt.plot(loss_values, color='navy', label = 'loss')
    plt.xlabel('Loss')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("DCAE_Loss")
    plt.close()
    z_ = torch.cat(z_)

    return net, z_


def evaluation(net, dataloader, device):

    """Testing the Deep SVDD model"""

    net = net.to(device)
    z_ = []
    scores = []
    label = []
    net.eval()
    print('Testing...')

    with torch.no_grad():

        for x, labels in dataloader:

            x = x.float().to(device)
            z = net.encode(x)
            x_hat = net.decode(z)
            z_.append(z.cpu().detach())
            score = (torch.mean((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
            scores.append(score.cpu().detach())
            label += list(zip(labels.cpu().data.numpy().tolist()))

    scores = torch.cat(scores)
    scores = np.array(scores)
    label = np.array(label)
    label = label.reshape(label.shape[0],1)

    ROC_value = (roc_auc_score(label, -scores) * 100)

    print('ROC AUC score: {:.2f}'.format(ROC_value))

    z_ = torch.cat(z_)

    return label, scores, ROC_value, z_