
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import sys
import importlib
import torch
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
import AnoGAN_cwru_model as AnoGAN
from sklearn.metrics import roc_auc_score
import time
importlib.reload(AnoGAN)

# Number of workers for dataloader
# Decide which device we want to run on

def Train_AnoGAN(args,train_loader, device):

    # Create the generator
    G = AnoGAN.Generator(latent_dim=args.latent_dim, num_gf= args.ngf, channels= args.channels, bias=False).to(device)
    G.apply(AnoGAN.weights_init)
    D = AnoGAN.Discriminator(num_df= args.ndf, channels= args.channels, bias=False).to(device)
    D.apply(AnoGAN.weights_init)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr = args.lr, weight_decay=1e-5, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr = args.lr, weight_decay=1e-5, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=15, T_mult=2)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, T_0=15, T_mult=2)

    img_list = []
    d_losses = []
    g_losses = []

    iters = 0

    D.train()
    G.train()

    steps_per_epoch = len(train_loader)
    start_time = time.time()

    for epoch in range(1,args.epochs + 1):

        for i, (images, _) in enumerate(train_loader):

            D.zero_grad()
            real_images = images.to(device)
            batch_num = images.size(0)
            optimizer_D.zero_grad()

            real_output = D(real_images)
            real_label = torch.ones_like(real_output, device=device)

            z = torch.randn(batch_num, args.latent_dim, 1, device=device)
            fake_images = G(z)
            fake_output = D(fake_images.detach())
            fake_label = torch.zeros_like(fake_output, device=device)

            real_lossD = criterion(real_output, real_label)
            fake_lossD = criterion(fake_output, fake_label)

            D_loss = real_lossD + fake_lossD
            D_loss.backward()
            optimizer_D.step()

            P_real = real_output.mean().item()  # Discriminator가 real image를 진짜라고 판별한 확률
            P_fake = fake_output.mean().item()  # Discriminator가 fake image를 진짜라고 판별한 확률

            for _ in range(3):

            ############# Update G network: maximize log(D(G(z))) ##############

                optimizer_G.zero_grad()
                fake_images = G(z)
                fake_output = D(fake_images)

                G_loss = criterion(fake_output, torch.ones_like(fake_output, device=device))
                G_loss.backward()
                optimizer_G.step()

            scheduler_D.step(epoch + i * steps_per_epoch)
            scheduler_G.step(epoch + i * steps_per_epoch)

            d_losses.append(D_loss.item())
            g_losses.append(G_loss.item())

            iters += 1

            torch.save(D.state_dict(), args.save_path + './all.tar')
            torch.save(G.state_dict(), args.save_path + './all.tar')

        if epoch % 5 == 0:

            print(f'Epoch {epoch}/{args.epochs} | D loss: {D_loss.item():.6f} | G loss: {G_loss.item():.6f} | P(real): {P_real:.4f} | P(fake): {P_fake:.4f}')
            img_list.append(G(z).detach().cpu().numpy())

    test_time = time.time() - start_time
    print('Testing_time : {:.3f}'.format(test_time))

    return D, G, test_time

def Test_AnoGAN(args, D, G,test_loader, device):

    G.eval()
    D.eval()

    z = torch.randn(1, args.latent_dim, 1, device=device, requires_grad=True)

    optimizer_z = torch.optim.Adam([z], lr = args.lr)

    latent_space = []
    score = []
    start_time = time.time()

    for i, (images, _) in enumerate(test_loader):

        real_images = images.to(device)
        print(f'image{i+1}')

        # 501 mean epoch for testing phase

        for step in range(501):

            generated_images = G(z)
            optimizer_z.zero_grad()

            real_features = D.forward_features(real_images)
            generated_features = D.forward_features(generated_images)

            residual_loss = torch.mean((generated_images - real_images)**2, dim = tuple(range(1,generated_images.dim())))
            discriminator_loss = torch.mean((generated_features - real_features)**2, dim = tuple(range(1,generated_features.dim())))

            anomaly_loss = ((1 - args.l) * residual_loss) + (args.l * discriminator_loss)

            anomaly_loss.backward(anomaly_loss,retain_graph = True)
            optimizer_z.step()

            if step % 5 == 0:

                print('Anomaly_loss : {:.3f}', torch.mean(anomaly_loss))

            if step == 10:

                score.append(anomaly_loss.cpu().detach())
                latent_space.append(z.cpu().data.numpy())

        test_time = time.time() - start_time

        print('Testing_time : {:.3f}', test_time)

    return score

def AnoGAN_ROC_Value(score, test_loader):

    label = test_loader.dataset.label
    labels = np.array(label)

    score = torch.cat(score)
    score = np.array(score)

    ROC_value = (roc_auc_score(labels, -score) * 100)
    print('ROC AUC score: {:.2f}'.format(ROC_value))

    return ROC_value
