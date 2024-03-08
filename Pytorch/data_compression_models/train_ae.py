import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder import Autoencoder
from utils import get_accuracy, get_reconstruction, plot_latent
import config_ae
# from dataset import MyImageFolder


def train(loader, model, optimizer, device):
    model.train()
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        # to device
        x = data.to(device)

        # forward
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss


def main():
    # load dataset
    train_data = datasets.MNIST(root="dataset/", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="dataset/", train=False, download=True, transform=transforms.ToTensor())
    # add data to dataloader
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    # configure model
    model = Autoencoder(input_dims=784, hidden_dims=512, latent_dims=20).to(config_ae.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config_ae.LEARNING_RATE)

    # train model
    train_loss = []
    for epoch in range(config_ae.NUM_EPOCHS):
        print(f"Processing epoch num: {epoch}")
        trained_ae, loss = train(train_loader, model, optimizer, config_ae.DEVICE)

        # keep track of losses
        train_loss.append(loss)

        # get_accuracy(train_loader, model.classifier, config_ae.BATCH_SIZE)
    plot_latent(trained_ae, train_loader, config_ae.DEVICE)
    print("here")


if __name__ == "__main__":
    main()