import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import os
from PIL import Image
from tqdm import tqdm

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(
        2 * math.pi) * log_stds.size(-1)
    # tanh による確率密度の変化を修正する．
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    acts = means + noises * stds
    return acts


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, inputs, size=256):
        ans = inputs.view(inputs.size(0), size, 3, 8)
        return ans



class Encoder(nn.Module):
    def __init__(self, channels=3, h_dim=6144, z_dim=32):
        super(Encoder, self).__init__()
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, inputs):  # encode z. (input -> encoder -> parameterize -> z, means, log_stds)
        tmp = self.encoder(inputs)
        means, log_stds = self.fc1(tmp), F.softplus(self.fc2(tmp))
        z = reparameterize(means, log_stds)
        return z, means, log_stds


class Decoder(nn.Module):
    def __init__(self, channels=3, h_dim=6144, z_dim=32):
        super(Decoder, self).__init__()
        self.channels = channels
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(z_dim, h_dim)

    def forward(self, inputs):  # decode z. (input -> fc1 -> decoder -> ans)
        z = self.fc1(inputs)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, channels=3, z_dim=3):
        # encoder -> z ~ N(\mu, \sigma) -> decoder
        super(VAE, self).__init__()
        self.channels = channels
        self.z_dim = z_dim
        self.encoder = Encoder().to(dev)
        self.decoder = Decoder().to(dev)

    def forward(self, inputs):
        z, means, log_stds = self.encoder(inputs)
        z = self.decoder(z)
        return z, means, log_stds

    def calc_loss(self, inputs, zs, means, log_stds):
        kl_loss = (-0.5 * torch.sum((1.0 + log_stds - means.pow_(2) - log_stds.exp()), dim=0)).mean()
        # print("zs {}".format(zs.shape))
        # print("ipnuts {}".format(inputs.shape))
        zs = zs.view(-1, 38400)
        inputs = inputs.contiguous().view(-1, 38400)
        loss = F.binary_cross_entropy(zs, inputs, reduction='sum')
        return loss + 5.0 * kl_loss


def train_vae(vae, epochs, train_datas):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for epoch in range(epochs):
        losses = []
        for data in tqdm(train_datas):
            images = data.to(dev)
            optimizer.zero_grad()
            zs, means, log_stds = vae(images)
            loss = vae.calc_loss(images, zs, means, log_stds)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print("epoch{}: average loss {}".format(epoch, np.array(losses).mean()))
        torch.save(vae.state_dict(), './vae.torch', _use_new_zipfile_serialization=False)


def load_pictures():
    pic_dir = "/home/emile/Documents/Code/RL_car/train_data/pictures"
    file_name = "_cam-image_array_.jpg"
    num_file = sum(os.path.isfile(os.path.join(pic_dir, name)) for name in os.listdir(pic_dir))
    ans = []
    for index in tqdm(range(num_file)):
        path = pic_dir + "/" + str(index) + file_name
        img = np.array(Image.open(path).crop((0, 40, 160, 120)))
        im = torch.from_numpy(img.reshape((1, 80, 160, 3))).to(dev).permute(0, 3, 1, 2).float()
        ans.append(im)
    return ans


def main():
    vae = VAE()
    pics = load_pictures()
    train_vae(vae, 1000, pics)


if __name__ == "__main__":
    main()
