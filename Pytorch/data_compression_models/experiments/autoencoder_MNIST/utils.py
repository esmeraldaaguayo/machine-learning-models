import torch
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


def plot_reconstructed(trained_model, device, r0=(-5, 10), r1=(-10, 5), n=12):
    """
    Reconstructs inputs from arbitrary latent vectors
    """
    f, ax = plt.subplots()
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x,y]]).to(device)
            x_hat = trained_model.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig('plots/reconstructed_images.png')


def plot_training_loss(loss):
    """
    Track training loss.
    """
    f,ax = plt.subplots()
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('plots/training_loss.png')


def plot_latents(autoencoder, data, device, num_batches=100):
    """
    Visualize 2D latents.
    """
    f, ax = plt.subplots()
    for i, (x, y) in enumerate(data):
        x = x.to(device)
        x_2d = torch.flatten(x, start_dim=1)
        z = autoencoder.encoder(x_2d)
        z = z.to('cpu').detach().numpy()
        pc = plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i >= num_batches:
            plt.colorbar(pc)
            break
    plt.savefig('plots/latents.png')
