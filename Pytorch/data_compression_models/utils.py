import torch
import matplotlib.pyplot as plt


def get_accuracy(data_loader, model, batch_size, device):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    model.eval()
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            # to device
            x = x.to(device=device)
            y = y.to(device=device)

            scores = torch.sigmoid(model(x))
            predictions = (scores > 0.5).float()
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


def get_reconstruction():
    pass


def plot_latent(autoencoder, data, device, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
        plt.savefig('plots/latents.png')
