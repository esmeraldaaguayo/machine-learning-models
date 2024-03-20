import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_compression_models.models.variational_autoencoder import VariationalAutoencoder
from utils import plot_latents, plot_training_loss, plot_reconstructed, inference
import config_vae


def train(loader, model, optimizer, device):
    model.train()
    train_epoch_loss = 0
    train_epoch_reconst_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        # to device
        x = data.to(device)

        # flatten images into 2D tensors
        x_2d = torch.flatten(x, start_dim=1)

        optimizer.zero_grad()
        loss, reconst_loss = model.loss_function(x_2d)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss
        train_epoch_reconst_loss += reconst_loss

    train_epoch_loss /= len(loader.dataset)
    train_epoch_reconst_loss /= len(loader.dataset)

    return model, train_epoch_loss, train_epoch_reconst_loss


def main():
    # Set seed
    torch.manual_seed(0)

    # load dataset
    train_data = datasets.MNIST(root="../dataset/", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="../dataset/", train=False, download=True, transform=transforms.ToTensor())
    # add data to dataloader
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    # configure model
    model = VariationalAutoencoder(input_dims=28*28, hidden_dims=512, latent_dims=2).to(config_vae.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config_vae.LEARNING_RATE)

    # train model
    train_recon_losses = []
    train_total_losses = []
    for epoch in range(config_vae.NUM_EPOCHS):
        print(f"Processing epoch num: {epoch}")
        trained_vae, train_epoch_loss, train_epoch_reconst_loss = train(train_loader, model, optimizer, config_vae.DEVICE)

        # keep track of losses
        train_recon_losses.append(train_epoch_loss.cpu().detach().numpy())
        train_total_losses.append(train_epoch_reconst_loss.cpu().detach().numpy())

    # visualizations
    plot_training_loss(train_recon_losses, "reconstruction_loss_training")
    plot_training_loss(train_total_losses, "reconstruction_KL_divergence_loss_training")

    plot_latents(trained_vae, train_loader, config_vae.DEVICE)
    plot_reconstructed(trained_vae, config_vae.DEVICE, r0=(-3, 3), r1=(-3, 3))

    inference(digit=8, model=trained_vae, dataset=train_data, num_examples=2)


if __name__ == "__main__":
    main()