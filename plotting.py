import matplotlib.pyplot as plt
import torch



def plot_training_history(D_losses, G_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(D_losses, label="Discriminator loss")
    plt.plot(G_losses, label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline GAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")


def plot_distribution_generation(G, noise_dim, device, X_train, dir):
    G.eval()

    with torch.no_grad():
        z = torch.randn(5000, noise_dim, device=device)
        synth = G(z).cpu().numpy()


    real_cont = X_train[:, :6]
    fake_cont = synth[:, :6]

    for i in range(6):
        plt.figure(figsize=(5,3))
        
        plt.hist(
            real_cont[:, i],
            bins=30,
            alpha=0.5,
            label="Real"
        )
        
        plt.hist(
            fake_cont[:, i],
            bins=30,
            alpha=0.5,
            label="Fake"
        )
        
        plt.legend()
        plt.savefig(f"{dir}/feature_{i+1}.png")
