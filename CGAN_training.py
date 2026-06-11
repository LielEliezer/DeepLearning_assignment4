import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from models import Generator, Discriminator, ConditionalDiscriminator, ConditionalGenerator
from data_loaders import get_loaders
from preprocessing import process_data
from plotting import plot_training_history, plot_distribution_generation
from constants import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor, X_train, X_test, y_train, y_test = process_data(target_col="income", 
             continuous_cols=CONTINUOUS_COLS, 
            categorical_cols=CATEGORICAL_COLS
            )

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

cgan_dataset = TensorDataset(X_train_tensor, y_train_tensor)

cgan_loader = DataLoader(
    cgan_dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train.shape[1]
noise_dim = 128
condition_dim = 2

cG = ConditionalGenerator(noise_dim, condition_dim, input_dim).to(device)
cD = ConditionalDiscriminator(input_dim, condition_dim).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer_cG = torch.optim.Adam(
    cG.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

optimizer_cD = torch.optim.Adam(
    cD.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

num_epochs = 50

cG_losses = []
cD_losses = []

for epoch in range(num_epochs):
    epoch_cG_loss = 0
    epoch_cD_loss = 0

    for real_batch, y_batch in cgan_loader:
        real_batch = real_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = real_batch.size(0)

        real_labels = torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.0)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # -------------------------
        # Train Conditional Discriminator
        # -------------------------
        z = torch.randn(batch_size, noise_dim, device=device)

        fake_batch = cG(z, y_batch).detach()

        cD_real = cD(real_batch, y_batch)
        cD_fake = cD(fake_batch, y_batch)

        cD_loss_real = criterion(cD_real, real_labels)
        cD_loss_fake = criterion(cD_fake, fake_labels)

        cD_loss = cD_loss_real + cD_loss_fake

        optimizer_cD.zero_grad()
        cD_loss.backward()
        optimizer_cD.step()

        # -------------------------
        # Train Conditional Generator
        # -------------------------
        for _ in range(2):
            z = torch.randn(batch_size, noise_dim, device=device)

            fake_batch = cG(z, y_batch)

            cD_fake = cD(fake_batch, y_batch)

            cG_loss = criterion(cD_fake, real_labels)

            optimizer_cG.zero_grad()
            cG_loss.backward()
            optimizer_cG.step()

        epoch_cD_loss += cD_loss.item()
        epoch_cG_loss += cG_loss.item()

    epoch_cD_loss /= len(cgan_loader)
    epoch_cG_loss /= len(cgan_loader)

    cD_losses.append(epoch_cD_loss)
    cG_losses.append(epoch_cG_loss)

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            cD_real_score = torch.sigmoid(cD(real_batch, y_batch)).mean().item()
            cD_fake_score = torch.sigmoid(cD(fake_batch, y_batch)).mean().item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"cD_loss: {epoch_cD_loss:.4f} "
            f"cG_loss: {epoch_cG_loss:.4f} "
            f"cD(real): {cD_real_score:.3f} "
            f"cD(fake): {cD_fake_score:.3f}"
        )

torch.save(cG.state_dict(), "CGAN/cG_weights.pth")
torch.save(cD.state_dict(), "CGAN/cD_weights.pth")

plot_training_history(cD_losses, cG_losses, title="CGAN/CGAN_training")
