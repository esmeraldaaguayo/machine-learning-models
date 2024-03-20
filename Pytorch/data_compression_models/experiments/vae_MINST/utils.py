import torch
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torchvision.utils import save_image


def inference(digit, model, dataset, num_examples = 1):
    """
    Generates (num_examples) of a particular digit.
    Get learned mu and sigma for that digit representation.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encoder(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    learned_dist = torch.distributions.Normal(mu, sigma)
    for example in range(num_examples):
        z = learned_dist.rsample()
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"plots/generated_{digit}_ex{example}.png")
    return None


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


def plot_training_loss(loss, title):
    """
    Track training loss.
    """
    f,ax = plt.subplots()
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(f'plots/{title}_plot.png')


def plot_latents(autoencoder, data, device, num_batches=100):
    """
    Visualize 2D latents.
    """
    f, ax = plt.subplots()
    for i, (x, y) in enumerate(data):
        x = x.to(device)
        x_2d = torch.flatten(x, start_dim=1)

        mu, sigma = autoencoder.encoder(x_2d)
        # sampling with reparameterization trick
        learned_dist = torch.distributions.Normal(mu, sigma)
        z = learned_dist.rsample()

        z = z.to('cpu').detach().numpy()
        pc = plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i >= num_batches:
            plt.colorbar(pc)
            break
    plt.savefig('plots/latents.png')
