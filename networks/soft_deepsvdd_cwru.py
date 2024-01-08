
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import optim
import sys
import time
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
import utils
from Bar import Bar

class Signal_network(nn.Module):

    def __init__(self, z_dim = 16):
        super(Signal_network, self).__init__()

        self.pool = nn.MaxPool1d(4)
        # Conv2d(input channel, output channel, kernel_size) --> bias can affect model collapse problem (고려 X)
        self.conv1 = nn.Conv1d(1, 8, 5, bias=False, padding=2)
        # BatchNorm2d(num_features, eps = 1e-05, affine)
        self.bn1 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        # Linear(in_features, out_features, bias = True)
        self.fc1 = nn.Linear(16 * z_dim, z_dim, bias=False)


    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(x, negative_slope=0.2))
        x = x.view(x.size(0), -1)

        return self.fc1(x)

class Soft_Signal_autoencoder(nn.Module):

    # init 함수에서 각 레이어를 지정해줌
    def __init__(self, z_dim = 16):
        super(Soft_Signal_autoencoder, self).__init__()

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
        x = self.pool(F.leaky_relu(x,negative_slope= 0.2))
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


def pretrain(args,train_loader, device):

    model = Soft_Signal_autoencoder(args.latent_dim).to(device)
    model.apply(utils.weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_ae, weight_decay=args.weight_decay_ae)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones_ae, gamma=0.1)

    model.train()
    creterion = nn.MSELoss()

    for epoch in range(args.num_epochs_ae):

        total_loss = 0
        # train_loader --> mnist 학습 데이터
        for x, _ in Bar(train_loader):
            # input data --> x
            x = x.float().to(device)
            optimizer.zero_grad()
            # output data --> x_hat
            x_hat = model(x)
            reconst_loss = creterion(x, x_hat)
            reconst_loss.backward()
            optimizer.step()
            # item() --> Tensor의 scalar값을 반환함
            total_loss += reconst_loss.item()

        scheduler.step()
        print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss / len(train_loader)))

        save_weights_for_DeepSVDD(args, model, train_loader, device)

def set_c(model, train_loader, device, eps=0.1):

    """Initializing the center for the hypersphere"""
    # 현재 모델에 평가단계라는 것을 입력함
    model.eval()
    z_ = []
    # grad 연산을 사용하지 않음
    with torch.no_grad():
        # y label이 존재하지 않음 --> _로 처리함
        for x, _ in train_loader:
            x = x.float().to(device)
            # initial forward 단계
            z = model.encode(x)
            z_.append(z.detach())
    # 1*n --> n*1의 1차원의 형태로 변환해줌
    z_ = torch.cat(z_)
    c = torch.mean(z_, dim=0)
    # Numerical stability
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def save_weights_for_DeepSVDD(args, model, train_loader, device):

    """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""

    # set_c에서 정해진 c를 가져옴, model --> pretain autoencoder를 입력
    c = set_c(model, train_loader, device, eps=0.1)
    # Deep SVDD 모델 가져옴
    net = Signal_network(args.latent_dim).to(device)
    # 모델의 파라미터를 저장함 --> 가중치, 편향 ( Deep SVDD에서는 편향을 사용하지 않음)
    state_dict = model.state_dict()
    net.load_state_dict(state_dict, strict=False)
    torch.save({'center': c.cpu().data.numpy().tolist(),'net_dict': net.state_dict()}, args.save_path + './all.tar')

def get_radius(args, dist):

    """Optimally solve for radius R via the (1-nu)-quantile of distances."""

    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - args.nu)

def train(args, train_loader, device, R):

    net = Signal_network().to(device)

    start_time = time.time()

    # initial R = 0
    if args.pretrain == True:

        state_dict = torch.load(args.save_path + './all.tar')
        net.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(device)

    else:
        net.apply(utils.weights_init_normal)
        c = torch.randn(args.latent_dim).to(device)

    # Set optimizer (Adam optimizer for now)
    optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    # Set learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_milestones, gamma=0.1)

    net.train()
    z_ = []

    for epoch in range(args.n_epochs):

        scheduler.step()

        if epoch in args.lr_milestones:

            print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

        loss_epoch = 0.0
        n_batches = 0

        for data in train_loader:

            inputs, _ = data
            inputs = inputs.to(device)
            # Zero the network parameter gradients
            optimizer.zero_grad()
            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = net(inputs)
            z_.append(outputs.detach().cpu())
            dist = torch.sum((outputs - c) ** 2, dim=1)

            if args.objective == 'soft-boundary':
                scores = dist - R ** 2
                loss = R ** 2 + (1 / args.nu) * torch.mean((torch.max(torch.zeros_like(scores), scores)))
            else:
                loss = torch.mean(dist)

            loss.backward()
            optimizer.step()

            # Update hypersphere radius R on mini-batch distances
            if (args.objective == 'soft-boundary') and (epoch % args.warm_up_n_epochs == 0):
                R = torch.tensor(get_radius(args, dist), device = device)

            loss_epoch += loss.item()
            n_batches += 1

        print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}, Radius: {:.3f}'.format(epoch + 1, loss_epoch / len(train_loader), R))

    test_time = time.time() - start_time

    print("Testing Time ... {:.3f}", test_time)

    z_ = torch.cat(z_, dim=0)
    z_ = z_.detach().cpu().numpy()

    return net,c, R, z_, test_time

def test(args, net, c, test_loader, R, device):

    # Set device for network

    # Testing
    net.eval()

    idx_label_score = []
    start_time = time.time()
    z_ = []

    with torch.no_grad():

        for x, labels in test_loader:

            x = x.float().to(device)
            z = net(x)
            z_.append(z.detach().cpu())
            dist = torch.sum((z - c) ** 2, dim=1)
            # cpu에서 계산을 할당함

            if args.objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    print('Testing time: {:.2f}'.format(test_time))

    labels, scores = zip(*idx_label_score)

    labels = np.array(labels)
    scores = np.array(scores)


    ROC_Value = (roc_auc_score(labels, -scores) * 100)

    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, -scores) * 100))

    z_ = torch.cat(z_, dim=0)

    z_ = z_.detach().cpu().numpy()

    return labels, scores, ROC_Value, z_


