# Resource

- torchgfn Github repository: https://github.com/GFNOrg/torchgfn
- torchgfn Document: https://torchgfn.readthedocs.io/en/latest/autoapi/gfn/env/index.html
- torchgfn Tutorial: https://github.com/GFNOrg/torchgfn/blob/9c9e1af83afcb92527d8ab9b3664d6a65d518864/tutorials/notebooks/intro_gfn_smiley.ipynb

# GFNEvalS

## Limitations and Future Work

- A key limitation of GFNEvalS is the **cost** to compute the exact probability of sampling a terminal state under the GFlowNet. This can make evaluation difficult with large test sets and larger training runs. 
  - One direction to explore would be ways to approximate this quantity more efficiently, while still retaining the properties of GFNEvalS.
  - Or compute the sampling probability using Dynamic Programming.
- Construct a toy environment with discrete states
  - (Not Sure) To make our toy environment simple enough, we hope its state space doesn't grow exponentially? 
    - Maybe HyperGrid is enough for us, we can do some experiments.

## Pseudocode for reproducing GFNEvalS

```python
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP
from gfn.states import DiscreteStates

# 0 - Find Available GPU resource
device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1 - Define the environment
env = HyperGrid(ndim=4, height=8, R0=0.01)

# 2 - Define the neural network modules
module_PF = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
module_PB = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions - 1, trunk=module_PF.trunk)

# 3 - Define the estimators
pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

# 4 - Define the GFlowNet
gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)

# 5 - Define the sampler and optimizer
sampler = Sampler(estimator=pf_estimator)
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

# 6 - Train the GFlowNet
for i in (pbar := tqdm(range(1000))):
    trajectories = sampler.sample_trajectories(env=env, n=16)
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories).to(device)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        pbar.set_postfix({"loss": loss.item()})
        
        # 8 - Generate a test set and compute probabilities
n_test = 100  # Number of test trajectories
test_trajectories = sampler.sample_trajectories(env=env, n=n_test)

# Initialize lists to hold the probabilities and rewards
log_probs = []
log_rewards = []
memo = {}
# Calculate the log probability and log reward for each terminal state
for traj in test_trajectories:
    terminal_states = traj[-1].states
    reward = env.reward(terminal_state)
    log_reward = np.log(reward)
    # TODO
    log_prob = compute_log_probability(gfn, terminal_states, memo)
    log_probs.append(log_prob)
    log_rewards.append(log_reward)
    # 9 - Compute Spearman's Rank Correlation
spearman_corr, _ = spearmanr(log_probs, log_rewards)
print(f"Spearman's Rank Correlation (GFNEvalS): {spearman_corr}")
```

### Compute sampling probability

TODO: I don't know how to get Parents(s) in torchgfn yet.

```python
import torch
def compute_log_probability(gfn, state, memo={}):
    """
    Recursively computes the log of the sampling probability π_θ(s) for a given terminal state `state`
    in a GFlowNet `gfn` using torchgfn library.
    
    Args:
        gfn (GFlowNet): The GFlowNet model instance.
        state (States): The terminal state for which we want to compute log π_θ(s).
        memo (dict): A dictionary for memoization to store previously computed log probabilities.
        
    Returns:
        torch.Tensor: The log probability π_θ(s).
    """
    # Check if the result is already computed and stored in memo
    if state in memo:
        return memo[state]
    
    # Base case: if the state is the initial state, log π_θ(s_initial) = 0
    if state.is_initial_state.all():
        log_prob = torch.tensor(0.0, requires_grad=False)
        memo[state] = log_prob
        return log_prob
    
    # Recursive case: compute log π_θ(s) from parent states
    # TODO
    parent_states = get_parents(state)
    
    # Collect log-probabilities for each parent transition
    log_probs = []
    for parent_state in parent_states:
        # Forward transition probability in log form
        log_forward_prob = torch.log(gfn.get_forward_transition_probability(state, parent_state))
        
        # Recursively compute log π_θ(parent_state)
        log_parent_prob = compute_log_probability(gfn, parent_state, memo)
        
        # Compute the sum inside the exponent for this parent
        log_probs.append(log_forward_prob + log_parent_prob)
    
    # Sum of exponentiated log-probabilities (log-sum-exp trick for numerical stability)
    log_prob = torch.logsumexp(torch.stack(log_probs), dim=0)
    
    # Memoize and return
    memo[state] = log_prob
    return log_prob
```



