import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_compression_models.models.variational_autoencoder import VariationalAutoencoder
from utils import plot_latents, plot_training_loss, plot_reconstructed
import config_vae


def train(loader, model, optimizer, criterion, device):
    model.train()

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        # to device
        x = data.to(device)

        # flatten images into 2D tensors
        x_2d = torch.flatten(x, start_dim=1)

        # forward
        mu, sigma, x_hat = model(x_2d)

        loss_fn = nn.MSELoss(reduction="sum")
        loss = loss_fn(x_hat, x_2d)
        train_epoch_loss = loss.cpu().detach().numpy()


        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, train_epoch_loss


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
    # criterion = torch.nn.MSELoss()

    # train model
    recon_losses = []
    kl_losses = []
    losses = []
    for epoch in range(config_vae.NUM_EPOCHS):
        print(f"Processing epoch num: {epoch}")
        trained_vae, train_epoch_loss = train(train_loader, model, optimizer, criterion, config_vae.DEVICE)

        # keep track of losses
        train_loss.append(train_epoch_loss)

    # visualizations
    plot_training_loss(train_loss)
    plot_latents(trained_ae, train_loader, config_vae.DEVICE)
    plot_reconstructed(trained_ae, config_vae.DEVICE)


if __name__ == "__main__":
    main()