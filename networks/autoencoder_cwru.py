import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import optim
import sys
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
from Bar import Bar


class Autoencoder(nn.Module):

    def __init__(self, z_dim = 16):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim
        # Conv2d(input channel, output channel, kernel_size) --> bias can affect model collapse problem (고려 X)
        self.lin1 = nn.Linear(1024, 512, bias=False)
        # BatchNorm2d(num_features, eps = 1e-05, affine)
        self.lin2 = nn.Linear(512, 256, bias=False)
        # Linear(in_features, out_features, bias = True)
        self.fc1 = nn.Linear(256, 16, bias=False)

        # ConvTraspose2d(in_channels, out_channels, kernel_size)

        self.delin1 = nn.Linear(16, 256)
        self.delin2 = nn.Linear(256, 512)
        self.delin3 = nn.Linear(512, 1024)

    # init 함수에서 정의한 레이어중 encode 부분에서 사용할 layer를

    def encode(self, x):
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = F.leaky_relu(x)
        # x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, x):

        # the input tensor, scale_factor : multiplier for spatial size
        x = self.delin1(x)
        x = F.leaky_relu(x)
        x = self.delin2(x)
        x = F.leaky_relu(x)
        x = self.delin3(x)

        return torch.sigmoid(x)

    def forward(self, x):

        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat

def train(args, train_loader, device):

    """Training the Deep SVDD model"""
    net = Autoencoder().to(device)

    optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_milestones, gamma=0.1)

    net.train()
    creterion = nn.MSELoss()

    loss_values = []
    z_ = []

    for epoch in range(args.num_epochs):

        total_loss = 0

        for x, _ in Bar(train_loader):

            x = x.float().to(device)
            optimizer.zero_grad()
            z = net.encode(x)
            x_hat = net.decode(z)
            loss = creterion(x,x_hat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch == (args.num_epochs-1):

                z_.append(z.cpu().detach())

        scheduler.step()
        print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss / len(train_loader)))
        loss_values.append(total_loss/ len(train_loader))

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