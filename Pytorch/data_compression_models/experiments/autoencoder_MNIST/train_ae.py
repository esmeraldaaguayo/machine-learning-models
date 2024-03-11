import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_compression_models.models.autoencoder import Autoencoder
from utils import plot_latents, plot_training_loss, plot_reconstructed
import config_ae


def train(loader, model, optimizer, device):
    model.train()

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        # to device
        x = data.to(device)

        # flatten images into 2D tensors
        x_2d = torch.flatten(x, start_dim=1)

        # forward
        x_hat = model(x_2d)

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
    model = Autoencoder(input_dims=28*28, hidden_dims=512, latent_dims=2).to(config_ae.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config_ae.LEARNING_RATE)

    # train model
    train_loss = []
    for epoch in range(config_ae.NUM_EPOCHS):
        print(f"Processing epoch num: {epoch}")
        trained_ae, train_epoch_loss = train(train_loader, model, optimizer, config_ae.DEVICE)

        # keep track of losses
        train_loss.append(train_epoch_loss)

    # visualizations
    plot_training_loss(train_loss)
    plot_latents(trained_ae, train_loader, config_ae.DEVICE)
    plot_reconstructed(trained_ae, config_ae.DEVICE)


if __name__ == "__main__":
    main()