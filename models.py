import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)




class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, output_dim)
        )

    def forward(self, z, y):
        y_onehot = F.one_hot(y, num_classes=2).float()
        x = torch.cat([z, y_onehot], dim=1)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, x, y):
        y_onehot = F.one_hot(y, num_classes=2).float()
        x = torch.cat([x, y_onehot], dim=1)
        return self.net(x)