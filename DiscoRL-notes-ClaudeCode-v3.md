# DiscoRL Notes - Claude Code Analysis (v3)

This document provides detailed answers to questions about the DiscoRL (Discovering Reinforcement Learning) algorithm based on analysis of the source code and documentation.

**Notes:** this doc is write by ClaudeCode with Opus 4.5 model. 2026.01.12.

---

## Question 1: What is the neural network architecture of the 'Agent-network'?

The Agent network is a **Multi-Layer Perceptron (MLP)** implemented in `disco_rl/networks/nets.py`.

### Architecture Details:

**Source: `disco_rl/networks/nets.py` (lines 2092-2260 in merged file)**

```
Input: Observation embedding
    ↓
Hidden Layer 1: Linear(hidden_size) → ReLU
    ↓
Hidden Layer 2: Linear(hidden_size) → ReLU
    ↓
[Multiple Output Heads]
```

**Configuration Parameters:**  
- `num_hidden_layers`: Number of hidden layers (default: 2).  
- `hidden_size`: Size of hidden layers (default: 256).  
- `num_actions`: Number of discrete actions.  
- `num_bins`: Number of bins for distributional outputs (default: 601).  

**Output Heads:**  
1. **π (Policy)**: `Linear(num_actions)` → Categorical distribution over actions.  
2. **y (Observation prediction)**: `Linear(num_bins)` → Distributional prediction conditioned on observation.  
3. **z (Action prediction)**: `Linear(num_bins × num_actions)` → Distributional prediction conditioned on action.  
4. **q (Q-value)**: `Linear(num_bins × num_actions)` → Distributional Q-values.  
5. **p (Auxiliary policy)**: `Linear(num_actions × num_actions)` → Auxiliary policy predictions.  

The network uses **distributional outputs** with 601 bins spanning values from -300 to +300, enabling fine-grained value representations.

---

## Question 2: How to train the Agent-network?

The Agent network is trained through **supervised learning** to match targets produced by the Meta-network.

### Training Process:

**Source: `disco_rl/update_rules/disco.py` (lines 4100-4200 in merged file)**

1. **Collect Experience**: Agent interacts with environment, collecting rollouts (observations, actions, rewards)

2. **Forward Pass**: Agent network produces predictions (π, y, z, q, p)

3. **Get Meta-Targets**: Meta-network generates target distributions (π̂, ŷ, ẑ)

4. **Compute Loss**: KL divergence between agent predictions and meta-targets:
   ```
   L_agent = D_KL(π̂ || π) + D_KL(ŷ || y) + D_KL(ẑ || z)
   ```

5. **Gradient Computation**: Backpropagate through agent network only (meta-parameters are fixed)

6. **Parameter Update**: Apply gradients using Adam optimizer

### Key Code (from `disco_rl/agent.py` lines 1200-1350):

```python
def learner_step(self, rng, rollout, learner_state, agent_net_state,
                 update_rule_params, is_meta_training):
    # Get agent outputs
    agent_outputs = self.agent_network(rollout.observations)

    # Compute loss using update rule (calls meta-network internally)
    loss, grads = jax.value_and_grad(self.update_rule.agent_loss)(
        learner_state.params, rollout, update_rule_params
    )

    # Apply gradient updates
    updates, new_opt_state = self.optimizer.update(grads, learner_state.opt_state)
    new_params = optax.apply_updates(learner_state.params, updates)
```

---

## Question 3: How to produce the predictions (π, y, z) from Agent-network?

### Inputs for Agent Producing Predictions:

**Source: `disco_rl/networks/nets.py` and `disco_rl/agent.py`**

1. **Raw observation** from environment (e.g., pixel frames for Atari)
2. **Preprocessed observation** after transformation
3. **Action** (for action-conditioned outputs z, q, p)

### Outputs of Predictions:

| Output | Shape | Description |
|--------|-------|-------------|
| π | `[B, num_actions]` | Policy logits → action probabilities |
| y | `[B, num_bins]` | Observation-conditioned prediction distribution |
| z | `[B, num_actions, num_bins]` | Action-conditioned prediction distribution |
| q | `[B, num_actions, num_bins]` | Distributional Q-values |
| p | `[B, num_actions, num_actions]` | Auxiliary policy predictions |

### Observation Transformation:

**For Atari environments:**  
1. Convert RGB to grayscale.  
2. Resize to 84×84 pixels.  
3. Stack 4 consecutive frames (temporal context).  
4. Normalize pixel values to [0, 1].  

**For continuous control:**  
1. Normalize features using running statistics.  
2. Optional clipping of extreme values.  

**Code flow (from `disco_rl/agent.py`):**
```python
def actor_step(self, actor_params, rng, timestep, actor_state):
    # Transform observation
    obs = preprocess(timestep.observation)

    # Forward through network
    outputs = self.agent_network.apply(actor_params, obs)

    # Sample action from policy
    action = outputs.pi.sample(rng)

    return action, outputs
```

---

## Question 4: How to calculate loss between Agent rollouts and targets from Meta-network?

**Source: `disco_rl/update_rules/disco.py` (lines 4100-4200)**

The loss is computed as **KL divergence** between meta-network targets and agent predictions:

### Loss Formula:

```
L_total = L_π + L_y + L_z

Where:
L_π = D_KL(π̂ || π) = Σ π̂(a) log(π̂(a) / π(a))
L_y = D_KL(ŷ || y) = Σ ŷ(b) log(ŷ(b) / y(b))
L_z = D_KL(ẑ || z) = Σ ẑ(b) log(ẑ(b) / z(b))
```

### Implementation (from `disco_rl/update_rules/disco.py`):

```python
def agent_loss(self, agent_params, rollout, meta_params):
    # Get agent predictions
    agent_outputs = self.agent_net(rollout.observations, agent_params)

    # Get meta-network targets
    meta_targets = self.unroll_meta_net(rollout, agent_outputs, meta_params)

    # KL divergence losses
    pi_loss = kl_divergence(meta_targets.pi_hat, agent_outputs.pi)
    y_loss = kl_divergence(meta_targets.y_hat, agent_outputs.y)
    z_loss = kl_divergence(
        meta_targets.z_hat,
        agent_outputs.z[rollout.actions]  # Only for taken action
    )

    # Average over time and batch
    total_loss = jnp.mean(pi_loss + y_loss + z_loss)
    return total_loss
```

### Key Points:
- KL divergence measures how different agent predictions are from meta targets
- Only the z-prediction for the **taken action** contributes to loss
- Loss is averaged over timesteps and batch dimensions

---

## Question 5: How to calculate gradients of Agent-network?

**Source: `disco_rl/agent.py` (lines 1250-1300) and `disco_rl/update_rules/disco.py`**

Gradients are computed using **JAX automatic differentiation**.

### Gradient Computation:

```python
def compute_agent_gradients(agent_params, rollout, meta_params):
    # Define loss function that only depends on agent params
    def loss_fn(params):
        # Forward pass
        agent_outputs = agent_network(rollout.observations, params)

        # Get targets (meta_params are NOT differentiated through)
        meta_targets = jax.lax.stop_gradient(
            meta_network(rollout, agent_outputs, meta_params)
        )

        # Compute KL loss
        return agent_loss(agent_outputs, meta_targets)

    # Compute loss and gradients simultaneously
    loss, grads = jax.value_and_grad(loss_fn)(agent_params)

    return loss, grads
```

### Key Implementation Details:

1. **`jax.value_and_grad`**: Efficiently computes both loss value and gradients in one pass

2. **`stop_gradient`**: Prevents gradients from flowing through meta-network - only agent parameters are updated

3. **Gradient structure**: `grads` has same tree structure as `agent_params` (nested dictionary of arrays)

4. **Batch handling**: Gradients are automatically averaged over batch dimension

---

## Question 6: How to update the parameters of Agent-network?

**Source: `disco_rl/agent.py` (lines 1300-1350)**

Parameters are updated using the **Adam optimizer** from Optax.

### Update Process:

```python
def update_agent_params(learner_state, grads):
    # Get current optimizer state
    opt_state = learner_state.opt_state
    params = learner_state.params

    # Compute parameter updates using Adam
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # Apply updates: new_params = params + updates
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state
```

### Adam Optimizer Configuration:

**From code analysis:**  
- Learning rate: Configurable (typically 1e-4 to 1e-3).  
- β1 = 0.9 (first moment decay).  
- β2 = 0.999 (second moment decay).  
- ε = 1e-8 (numerical stability).  

### Update Formula (Adam):

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t          # First moment
v_t = β2 * v_{t-1} + (1 - β2) * g_t²         # Second moment
m̂_t = m_t / (1 - β1^t)                       # Bias correction
v̂_t = v_t / (1 - β2^t)                       # Bias correction
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        # Parameter update
```

---

## Question 7: What is the neural network architecture of the 'Meta-network'?

**Source: `disco_rl/networks/meta_nets.py` (lines 2261-2633 in merged file)**

The Meta-network is an **LSTM-based recurrent neural network** with action-invariant architecture.

### Architecture:

```
Inputs: [y, z, q, p, π, r, γ, a] (configurable subset)
    ↓
Input Projection: Linear(input_dim → lstm_size)
    ↓
LSTM Core: LSTMCell(lstm_size)
    ↓
[Output Heads]
    ├── π̂ head: Linear(lstm_size → num_actions)
    ├── ŷ head: Linear(lstm_size → num_bins)
    └── ẑ head: Linear(lstm_size → num_bins)
```

### Key Components:

1. **Action-Invariant Design**: Weights are shared across action dimensions to reduce parameter count and improve generalization.  

2. **LSTM Core**: Standard LSTM with forget gate, input gate, output gate.  
   - Hidden size: Configurable (e.g., 256 or 512).  
   - Processes temporal sequence of agent experiences.  

3. **Input Encoding**:  
   - Reward: One-hot encoded into bins.  
   - Action: One-hot encoded.  
   - Predictions (y, z, q, p): Probability distributions.  

### MetaLSTM (Per-Lifetime RNN):

**Source: `disco_rl/networks/meta_nets.py` (lines 2500-2633)**

Additional LSTM unrolled across **agent updates** (not timesteps):

```
Update 1 summary → MetaLSTM → hidden_1
Update 2 summary → MetaLSTM → hidden_2
...
Update N summary → MetaLSTM → hidden_N
```

This captures the learning dynamics of the agent across its lifetime.

---

## Question 8: How to train the Meta-network?

**Source: `disco_rl/value_fns/value_fn.py` and supplementary materials**

The Meta-network is trained using **policy gradient methods** to maximize agent lifetime returns.

### Training Process:

1. **Collect Agent Lifetimes**: Run multiple agents with current meta-parameters
   - Each agent trains for K updates in an environment
   - Collect trajectories and returns

2. **Compute Advantages**: Using V-trace or GAE
   ```
   A_t = r_t + γV(s_{t+1}) - V(s_t) + γλA_{t+1}
   ```

3. **Policy Gradient Loss**:
   ```
   L_meta = -E[A_t * log p(target_t | history_t)]
   ```

4. **Value Function Loss**: MSE between value predictions and returns

5. **Update Meta-Parameters**: Using Adam optimizer

### Meta-Optimization Objective:

```
max_φ E[Σ_t γ^t r_t | agent trained with meta-network φ]
```

Where φ are the meta-network parameters.

---

## Question 9: How to produce the targets (π̂, ŷ, ẑ) from Meta-network?

### Inputs for Meta-network:

**Source: `disco_rl/update_rules/disco.py` (lines 3950-4050)**

The meta-network input is constructed from:

| Input | Description | Shape |
|-------|-------------|-------|
| y | Agent's observation prediction | `[T, B, num_bins]` |
| z | Agent's action prediction | `[T, B, num_actions, num_bins]` |
| q | Agent's Q-values | `[T, B, num_actions, num_bins]` |
| p | Agent's auxiliary policy | `[T, B, num_actions, num_actions]` |
| π | Agent's policy | `[T, B, num_actions]` |
| r | Reward (one-hot encoded) | `[T, B, num_reward_bins]` |
| γ | Discount factor | `[T, B, 1]` |
| a | Action (one-hot) | `[T, B, num_actions]` |

### Input Construction Code:

```python
def get_input_option(self):
    # Configurable input selection
    return {
        'y': True,   # Include y predictions
        'z': True,   # Include z predictions
        'q': True,   # Include Q-values
        'p': True,   # Include aux policy
        'pi': True,  # Include policy
        'r': True,   # Include reward
        'gamma': True,  # Include discount
        'a': True    # Include action
    }
```

### Outputs:

| Output | Shape | Description |
|--------|-------|-------------|
| π̂ | `[T, B, num_actions]` | Target policy distribution |
| ŷ | `[T, B, num_bins]` | Target observation prediction |
| ẑ | `[T, B, num_bins]` | Target action prediction |

### Target Generation with Bootstrapping:

**Key insight**: Targets use **future predictions** bootstrapped to current timestep:

```python
def compute_targets(self, rollout, agent_outputs, meta_params):
    # LSTM processes sequence
    lstm_outputs, _ = self.meta_lstm(inputs, initial_state)

    # Generate raw targets
    pi_hat_raw = self.pi_head(lstm_outputs)
    y_hat_raw = self.y_head(lstm_outputs)
    z_hat_raw = self.z_head(lstm_outputs)

    # Bootstrap: shift future predictions to current timestep
    # This creates temporal credit assignment
    y_hat = bootstrap_predictions(y_hat_raw, discounts)
    z_hat = bootstrap_predictions(z_hat_raw, discounts)

    return MetaTargets(pi_hat, y_hat, z_hat)
```

---

## Question 10: How to calculate loss of Meta-network?

**Source: `disco_rl/value_fns/value_fn.py` (lines 3500-3600)**

The meta-network loss combines **policy gradient loss** and **value function loss**.

### Policy Gradient Loss:

```python
def meta_policy_loss(meta_params, trajectories, advantages):
    total_loss = 0

    for traj in trajectories:
        # Log probability of meta-network's decisions
        meta_log_probs = compute_meta_log_probs(meta_params, traj)

        # Weighted by advantages (variance reduction)
        policy_loss = -jnp.mean(advantages * meta_log_probs)
        total_loss += policy_loss

    return total_loss / len(trajectories)
```

### Value Function Loss:

```python
def value_loss(value_params, trajectories):
    # Compute value predictions
    value_preds = value_network(trajectories.states, value_params)

    # Compute targets (discounted returns or TD targets)
    value_targets = compute_value_targets(trajectories)

    # MSE loss
    return jnp.mean((value_preds - value_targets) ** 2)
```

### Combined Loss:

```
L_total = L_policy + c_v * L_value + c_e * L_entropy

Where:
- c_v: Value loss coefficient (e.g., 0.5)
- c_e: Entropy bonus coefficient (e.g., 0.01)
```

---

## Question 11: How to calculate gradient of Meta-network?

**Source: Meta-training loop analysis**

Gradients are computed through the **entire agent training process** using JAX autodiff.

### Gradient Computation:

```python
def compute_meta_gradients(meta_params, value_params, environments):
    def meta_objective(params):
        total_return = 0

        for env in environments:
            # Initialize fresh agent
            agent_params = init_agent()

            # Train agent with meta-network (differentiable)
            for update in range(num_updates):
                rollout = collect_rollout(env, agent_params)

                # Agent update using meta-targets
                agent_params = agent_update(
                    agent_params, rollout, params  # params = meta_params
                )

                total_return += sum(rollout.rewards)

        return -total_return  # Negative for minimization

    # Compute gradients w.r.t. meta-parameters
    loss, grads = jax.value_and_grad(meta_objective)(meta_params)
    return loss, grads
```

### Key Technical Details:

1. **Through-time differentiation**: Gradients flow through entire agent training trajectory

2. **Stop gradients for stability**: Some terms use `stop_gradient` to prevent gradient explosion:
   ```python
   targets = jax.lax.stop_gradient(meta_targets)  # For some terms
   ```

3. **Advantage normalization**: Advantages normalized for stable gradients:
   ```python
   advantages = (advantages - mean(advantages)) / (std(advantages) + eps)
   ```

---

## Question 12: How to update the parameters of Meta-network?

**Source: Meta-optimization code analysis**

Meta-parameters are updated using **Adam optimizer** with optional gradient clipping.

### Update Process:

```python
def update_meta_params(meta_params, meta_opt_state, grads):
    # Optional: Clip gradients for stability
    grads = clip_gradients(grads, max_norm=1.0)

    # Compute Adam updates
    updates, new_opt_state = meta_optimizer.update(
        grads,
        meta_opt_state,
        meta_params
    )

    # Apply updates
    new_params = optax.apply_updates(meta_params, updates)

    return new_params, new_opt_state
```

### Optimizer Configuration:

```python
meta_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=1e-4)    # Adam optimizer
)
```

### Learning Rate Schedule:

Often uses learning rate warmup and decay:
```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-4,
    warmup_steps=1000,
    decay_steps=100000
)
```

---

## Question 13: How to calculate 'Advantage estimates'?

**Source: `disco_rl/value_fns/value_utils.py` (lines 2900-3100)**

Advantages are computed using **V-trace** for off-policy correction.

### V-trace Advantage Formula:

```
A_t = ρ_t * (r_t + γ * V(s_{t+1}) - V(s_t)) + γ * c_t * A_{t+1}

Where:
ρ_t = min(ρ̄, π(a_t|s_t) / μ(a_t|s_t))  # Truncated IS ratio
c_t = min(c̄, π(a_t|s_t) / μ(a_t|s_t))  # Trace cutting coefficient
```

### Implementation:

```python
def compute_advantages_vtrace(rollout, value_estimates, config):
    T = rollout.length
    advantages = jnp.zeros([T])

    # Importance sampling ratios
    log_rhos = rollout.target_log_probs - rollout.behavior_log_probs
    rhos = jnp.clip(jnp.exp(log_rhos), max=config.rho_max)  # ρ̄ = 1.0
    cs = jnp.clip(jnp.exp(log_rhos), max=config.c_max)      # c̄ = 1.0

    # Backward pass
    next_advantage = 0.0
    for t in reversed(range(T)):
        # TD error
        td_error = (
            rollout.rewards[t] +
            rollout.discounts[t] * value_estimates[t + 1] -
            value_estimates[t]
        )

        # V-trace advantage
        advantages[t] = rhos[t] * td_error + rollout.discounts[t] * cs[t] * next_advantage
        next_advantage = advantages[t]

    return advantages
```

### Alternative: GAE (Generalized Advantage Estimation)

```python
def compute_advantages_gae(rollout, value_estimates, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0

    for t in reversed(range(len(rollout))):
        delta = rollout.rewards[t] + gamma * value_estimates[t+1] - value_estimates[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    return jnp.array(advantages)
```

---

## Question 14: What is the purpose of 'Advantage estimates'? Where are they used? When are they used?

### Purpose:

1. **Variance Reduction**: Advantages center rewards around a baseline (value function), reducing gradient variance
   ```
   Var[A_t] << Var[R_t]  (typically)
   ```

2. **Credit Assignment**: Separates "how good was this action" from "how good was this state"
   ```
   A(s,a) = Q(s,a) - V(s)
   ```

3. **Bias-Variance Trade-off**: Configurable λ parameter balances:
   - λ = 0: Low variance, high bias (TD(0))
   - λ = 1: High variance, low bias (Monte Carlo)

### Where Used:

1. **Meta-network Policy Gradient** (Primary use):
   ```python
   # disco_rl/value_fns/value_fn.py
   meta_loss = -advantages * log_probs  # Policy gradient with baseline
   ```

2. **Agent Training** (if using policy gradient):
   ```python
   # Some discovered algorithms may use advantages
   agent_policy_loss = -advantages * agent_log_probs
   ```

### When Used:

1. **During meta-training**:
   - Computed after each agent lifetime
   - Used to update meta-network parameters

2. **Temporal scope**:
   - Computed for entire episode/trajectory
   - Backward pass from final timestep to first

3. **Update frequency**:
   - Recomputed after each meta-update
   - Value function updated to improve estimates

### Code Location:

**Source: `disco_rl/value_fns/value_utils.py`**
```python
def estimate_values(rollout, value_fn_params, config):
    # Get value predictions
    values = value_fn.apply(value_fn_params, rollout.observations)

    # Compute advantages
    advantages = compute_advantages_vtrace(rollout, values, config)

    return values, advantages
```

---

## Question 15: What is the function of the 'Value Function' in file value_fn.py?

**Source: `disco_rl/value_fns/value_fn.py` (lines 3418-3618)**

### Primary Function:

The Value Function serves as a **baseline for variance reduction** in meta-learning policy gradients.

### Why is a Value Function Necessary?

1. **High Variance Problem**: Raw returns have high variance
   ```
   Var[∇log π(a|s) * R] >> Var[∇log π(a|s) * A]
   ```

2. **Unbiased Baseline**: Subtracting V(s) doesn't change expected gradient but reduces variance:
   ```
   E[∇log π(a|s) * (R - V(s))] = E[∇log π(a|s) * R]  (unbiased)
   ```

3. **Faster Learning**: Lower variance → more stable gradients → faster convergence

### Implementation (from `value_fn.py`):

```python
class ValueFunction:
    def __init__(self, config):
        self.network = hk.nets.MLP([256, 256, 1])
        self.optimizer = optax.adam(config.value_lr)

    def get_value_outs(self, params, rollout):
        """Compute value estimates for states in rollout"""
        values = self.network.apply(params, rollout.observations)
        return values

    def update(self, params, opt_state, rollout, targets):
        """Update value function towards targets"""
        def loss_fn(p):
            preds = self.network.apply(p, rollout.observations)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss
```

### Why Does the Value Function Change?

1. **Non-stationary Targets**: As meta-network improves, agent behavior changes → value distribution shifts

2. **Environment Distribution**: Training across multiple environments requires adaptive value estimates

3. **Agent Learning Dynamics**: Value function must track agent's improving performance over lifetime

### What is Being Modified?

When the value function changes, the following are updated:

1. **Network Weights**:
   - Input layer weights mapping observations to hidden features
   - Hidden layer weights
   - Output layer weights producing value predictions

2. **Optimizer State**:
   - Adam momentum terms (first moment estimates)
   - Adam velocity terms (second moment estimates)

### Code for Value Function Update:

**Source: `disco_rl/value_fns/value_fn.py` (lines 3550-3600)**

```python
def value_loss_from_target(value_params, rollout, targets):
    """
    Compute MSE loss between value predictions and targets

    Args:
        value_params: Value network parameters
        rollout: Experience data
        targets: Target values (e.g., discounted returns, TD targets)

    Returns:
        Scalar loss value
    """
    predictions = value_network.apply(value_params, rollout.observations)
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss
```

### Summary:

| Aspect | Description |
|--------|-------------|
| **Purpose** | Variance reduction for policy gradients |
| **Inputs** | Observations (states) from agent trajectory |
| **Outputs** | Scalar value estimates V(s) |
| **Training** | MSE loss against TD or Monte Carlo targets |
| **Usage** | Computing advantages: A = R - V(s) |
| **Updates** | After each batch of agent lifetimes |

---

## Summary Table

| Component | Architecture | Training Method | Key Function |
|-----------|--------------|-----------------|--------------|
| Agent Network | MLP (2 hidden layers) | KL to meta-targets | Produce π, y, z, q, p |
| Meta-Network | LSTM | Policy gradient | Generate targets π̂, ŷ, ẑ |
| Meta-RNN | LSTM | End-to-end | Track learning dynamics |
| Value Function | MLP | MSE regression | Variance reduction |

---

*Generated by Claude Code (Opus 4.5) - January 2026*
