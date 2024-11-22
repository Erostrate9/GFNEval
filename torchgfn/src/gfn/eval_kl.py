'''
  Empirical KL divergence calculator
  By Yue Zhang, Nov 15, 2024
'''
import torch
torch.set_default_dtype(torch.float64)
from torch import Tensor
from torch import nn
from torch import optim

# Simple learnable function to approximate KL-divergence
class PhiFunction(nn.Module):
    def __init__(self, input_size, layer_size) -> None:
        super(PhiFunction, self).__init__()
        self.linear1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        # x = x.view(-1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

def calc_KL_using_model(model, samples_p, samples_q, no_grad=False):
    # Compute f(x) for samples from P and Q
    if no_grad:
        with torch.no_grad():
            f_p = model(samples_p)  # Output shape: [batch_size, 1]
            f_q = model(samples_q)  # Output shape: [batch_size, 1]
    else:
        f_p = model(samples_p)  # Output shape: [batch_size, 1]
        f_q = model(samples_q)  # Output shape: [batch_size, 1]

    # Compute the terms of the formula
    term_p = torch.mean(f_p)  # Expectation over P: E_P[f]
    term_q = torch.log(torch.mean(torch.exp(torch.clamp(f_q, max=695, min=-695))))  # Log of expectation over Q: log(E_Q[e^f])

    # KL divergence
    kl_div = term_p - term_q
    return kl_div

def compute_KL(p_star_sample : Tensor, p_hat_sample : Tensor,
                layer_size=128, num_epochs=200, lr=0.001, show_progress=False, 
                device='cuda'):
    # Ensure both samples have the same shape
    assert p_star_sample[0].shape == p_hat_sample[0].shape
    input_size = p_star_sample[0].numel()
    # print(input_size)
    # The function to learn
    phi = PhiFunction(input_size=input_size, layer_size=layer_size)
    phi = phi.to(device)
    optimizer = optim.Adam(phi.parameters(), lr=lr)
    # Learn the model
    for epoch in range(num_epochs):
        # Compute KL divergence
        kl_div = calc_KL_using_model(phi, p_star_sample, p_hat_sample)

        # The loss is the negation of the KL-divergence
        loss = -kl_div

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 and show_progress:
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
