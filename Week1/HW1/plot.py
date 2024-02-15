from matplotlib import pyplot as plt
import torch
import numpy as np

def plot_gelu(x):
    # plot the derivative of GELU
    # x = torch.linspace(-5, 5, 100, requires_grad=True)
    x = torch.tensor(x, requires_grad=True)
    y = torch.nn.functional.gelu(x)
    y.backward(torch.ones_like(x))
    return x.grad.detach().numpy()

def plot_leaky_relu(x):
    # plot the derivative of LeakyReLU
    x = torch.tensor(x, requires_grad=True)
    # x = torch.linspace(-5, 5, 100, requires_grad=True)
    y = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    y.backward(torch.ones_like(x))
    return x.grad.detach().numpy()

def plot_relu(x):
    # x = torch.linspace(-5, 5, 100, requires_grad=True)
    x = torch.tensor(x, requires_grad=True)
    y = torch.nn.functional.relu(x)
    y.backward(torch.ones_like(x))
    return x.grad.detach().numpy()

def plot_tanh(x):
    # x = torch.linspace(-5, 5, 100, requires_grad=True)
    x = torch.tensor(x, requires_grad=True)
    y = torch.tanh(x)
    y.backward(torch.ones_like(x))
    return x.grad.detach().numpy()

def plot():
    # x = torch.linspace(-5, 5, 100, requires_grad=True)
    x = np.linspace(-5, 5, 100)
    y1 = plot_gelu(x)
    y2 = plot_leaky_relu(x)
    y3 = plot_relu(x)
    y4 = plot_tanh(x)
    fig_width, fig_height = 10, 6
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(x, y1, label='derivative for GELU')
    plt.plot(x, y2, label='derivative for LeakyReLU')
    plt.plot(x, y3, label='derivative for ReLU')
    plt.plot(x, y4, label='derivative for tanh')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot()