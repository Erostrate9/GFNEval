import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch import optim

class PhiFunction(nn.Module):
    def __init__(self, input_size, layer_size) -> None:
        super(PhiFunction, self).__init__()
        self.linear1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(layer_size, 1)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

def calc_KL_using_model(model, samples_p, samples_q, clamp_min=-600, clamp_max=600, no_grad=False):
    """
    Compute the KL divergence using Donsker-Varadhan's variational formula.

    Args:
        model (nn.Module): The neural network approximating `f`.
        samples_p (torch.Tensor): Samples from distribution P.
        samples_q (torch.Tensor): Samples from distribution Q.

    Returns:
        torch.Tensor: The estimated KL divergence.
    """
    if no_grad:
        with torch.no_grad():
            f_p = model(samples_p)
            f_q = model(samples_q)
    else:
        f_p = model(samples_p)
        f_q = model(samples_q)

    # Compute the terms of the formula
    term_p = torch.mean(f_p)
    term_q = torch.log(torch.mean(torch.exp(torch.clamp(f_q, min=clamp_min, max=clamp_max))))  # Clamping
    
    kl_div = term_p - term_q
    return kl_div

def compute_KL(p_star_sample: Tensor, p_hat_sample: Tensor, layer_size=128, num_epochs=200, lr=0.001):
    """
    Function to learn the KL divergence from two samples.

    Args:
        p_star_sample (Tensor): Samples from the true distribution P.
        p_hat_sample (Tensor): Samples from the approximate distribution Q.
        layer_size (int): Size of the hidden layer for the neural network.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        torch.Tensor: The learned KL divergence value.
        PhiFunction: The trained model.
    """
    assert p_star_sample.shape == p_hat_sample.shape, "Samples must have the same shape"
    
    input_size = p_star_sample.numel()
    phi = PhiFunction(input_size=input_size, layer_size=layer_size)
    optimizer = optim.Adam(phi.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        kl_div = calc_KL_using_model(phi, p_star_sample, p_hat_sample)

        # Loss is the negative of the KL divergence
        loss = -kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 100 epochs (customizable)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, KL Divergence Estimate: {kl_div:.4f}")

    return kl_div, phi


# Test the performance
if __name__ == "__main__":
    batch_size = 10068
    n_features = 17
    num_epochs = 200

    # KL-divergence from two samples of the same normal distribution
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.randn(batch_size, n_features)

    # Compute empirical KL
    kl, phi = compute_KL(sampleA, sampleB, num_epochs=num_epochs)
    print(f"Empirical KL from two samples of the same normal distribution = {kl.item():.4f}")

    # Sample from the same distribution and calculate KL
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.randn(batch_size, n_features)
    kl = calc_KL_using_model(phi, sampleA, sampleB, no_grad=True)
    print(f"Empirical KL from another two samples of the same distribution = {kl.item():.4f}")

    # KL-divergence from two different distributions
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.rand(batch_size, n_features)
    kl, phi = compute_KL(sampleA, sampleB, num_epochs=num_epochs)
    print(f"Empirical KL from two samples of different distributions = {kl.item():.4f}")

    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.rand(batch_size, n_features)
    kl = calc_KL_using_model(phi, sampleA, sampleB, no_grad=True)
    print(f"Empirical KL from another two samples = {kl.item():.4f}")
