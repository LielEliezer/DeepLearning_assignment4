import torch
import torch.nn as nn

from models import Generator, Discriminator
from data_loaders import get_loaders
from preprocessing import process_data
from plotting import plot_training_history, plot_distribution_generation
from constants import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(NOISE_DIM, INPUT_DIM).to(device)
D = Discriminator(INPUT_DIM).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(
    G.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    D.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

preprocessor, X_train, X_test, y_train, y_test = process_data(target_col="income", 
             continuous_cols=CONTINUOUS_COLS, 
            categorical_cols=CATEGORICAL_COLS
            )

train_loader = get_loaders(X_train)
num_epochs = 50

G_losses = []
D_losses = []

for epoch in range(num_epochs):
    epoch_G_loss = 0
    epoch_D_loss = 0

    for (real_batch,) in train_loader:
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # -----------------
        # Train Discriminator
        # -----------------
        z = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_batch = G(z).detach()

        D_real = D(real_batch)
        D_fake = D(fake_batch)

        D_loss_real = criterion(D_real, real_labels)
        D_loss_fake = criterion(D_fake, fake_labels)
        D_loss = D_loss_real + D_loss_fake

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        z = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_batch = G(z)

        D_fake = D(fake_batch)

        G_loss = criterion(D_fake, real_labels)

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        epoch_D_loss += D_loss.item()
        epoch_G_loss += G_loss.item()

    epoch_D_loss /= len(train_loader)
    epoch_G_loss /= len(train_loader)

    D_losses.append(epoch_D_loss)
    G_losses.append(epoch_G_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"D_loss: {epoch_D_loss:.4f} "
            f"G_loss: {epoch_G_loss:.4f}"
        )
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            D_real_score = torch.sigmoid(D(real_batch)).mean().item()
            D_fake_score = torch.sigmoid(D(fake_batch)).mean().item()

        print(
            f"D(real)={D_real_score:.3f} "
            f"D(fake)={D_fake_score:.3f}"
        )

torch.save(G.state_dict(), "GAN/G_weights.pth")
torch.save(D.state_dict(), "GAN/D_weights.pth")
plot_training_history(D_losses, G_losses, title="GAN/GAN_training")
plot_distribution_generation(G, NOISE_DIM, device, X_train, dir="GAN")