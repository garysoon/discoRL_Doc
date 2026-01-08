# discoRL的代码理解-claudeCode-v2  
.  
# DiscoRL Meta-Training Process Analysis

## Overview

DiscoRL (Discovered Reinforcement Learning) is a meta-learning approach that discovers RL algorithms by learning a meta-network that generates learning targets for an agent. This document explains how the Agent and Meta-network are trained in the `meta_train.ipynb` notebook.

## Two-Level Optimization Structure

The training involves a nested optimization process:

### Inner Loop (Agent Optimization)
- **Objective**: Update agent parameters θ to minimize loss based on targets from meta-network
- **Parameters being updated**: Agent network parameters (θ)
- **Data**: Training rollouts from environment

### Outer Loop (Meta-Optimization)
- **Objective**: Update meta-network parameters η to improve agent's validation performance
- **Parameters being updated**: Meta-network parameters (η)
- **Data**: Validation rollouts after agent updates

## Key Components

### 1. MetaTrainAgent Class (meta_train.ipynb:cell-5)

The `MetaTrainAgent` class bundles together:
- **agent**: The learning agent (agent_lib.Agent)
- **value_fn**: A separate value function for computing advantages (used only in meta-training)
- **env**: The environment (Catch game in this example)

Key methods:
- `init_state()`: Initializes all states (learner, actor, value function, environment)
- `unroll_actor()`: Collects trajectories by running the agent in the environment

### 2. Agent Training - Inner Loop (agent.py:248-316)

The `learner_step()` method implements one agent parameter update:

```python
def learner_step(self, rng, rollout, learner_state, agent_net_state,
                 update_rule_params, is_meta_training):
```

**Step-by-step process:**

1. **Unroll agent network** on the rollout to get agent outputs (logits, y, z, q)

2. **Apply meta-network** via `update_rule.unroll_meta_net()`:
   - Input: Agent's current parameters, rollout data
   - Output: Targets (pi_hat, y_hat, z_hat) from meta-network
   - Location: disco.py:112-208

3. **Compute agent loss** via `self._loss()`:
   - KL divergence between agent outputs and meta-network targets
   - Loss = pi_cost * KL(pi_hat || logits) + y_cost * KL(y_hat || y) + z_cost * KL(z_hat || z)
   - Location: agent.py:200-246, disco.py:210-294

4. **Compute gradients**: `grads = jax.grad(self._loss)(params, ...)`

5. **Update agent parameters**:
   - Apply optimizer transformation
   - new_params = optax.apply_updates(params, updates)

### 3. Meta-Network Architecture (meta_nets.py:45-158)

The meta-network is an LSTM-based architecture with two components:

#### Per-Trajectory LSTM
- Processes each rollout independently
- **Unrolls in reverse** (for bootstrapping from future states)
- Hidden size: 256 (configurable)
- Input: Transformed features from rollout (rewards, observations, policies, values, etc.)

#### Meta-LSTM (MetaLSTM class: meta_nets.py:160-227)
- Processes information **across the agent's entire lifetime**
- Maintains persistent state across rollouts
- Updates once per rollout after processing all timesteps
- Hidden size: 128 (configurable)

**Meta-Network Outputs:**
- **pi_hat**: Policy target (logits) for updating agent's policy
- **y_hat**: Target for observation-conditional prediction head
- **z_hat**: Target for action-conditional prediction head

**Key architectural features:**
1. **Bootstrapping**: Per-trajectory LSTM runs backwards to use future information
2. **Multiplicative interaction**: Combines per-trajectory features with meta-LSTM state
3. **Rich input features**: Over 15 different transformed inputs (see disco.py:326-428)

### 4. Meta-Network Target Generation (disco.py:112-208)

The `unroll_meta_net()` method generates learning targets:

**Process:**

1. **Compute target policy** by running agent with target parameters (exponentially moving average of current parameters)

2. **Compute value estimates** using Q-function and advantages:
   - Q-values from agent's q-head
   - TD-lambda returns with discount factor
   - Normalized advantages using exponential moving average

3. **Prepare meta-network inputs**:
   - Current and behavior policies
   - Rewards (transformed with sign-log)
   - Values and advantages
   - Agent's predictions (y, z)
   - Target network's outputs

4. **Apply meta-network** (LSTM) to generate targets (pi_hat, y_hat, z_hat)

5. **Update meta-state**:
   - RNN hidden state
   - Advantage and TD exponential moving averages
   - Target parameters (τ = 0.9 by default)

### 5. Meta-Gradient Calculation (meta_train.ipynb:cell-6)

The `calculate_meta_gradient()` function implements the outer loop:

**Input:**
- `update_rule_params`: Meta-network parameters η
- `agent_state`: Current agent state
- `train_rollouts`: Multiple training rollouts for inner updates
- `valid_rollout`: Validation rollout for meta-loss computation

**Process:**

#### Step 1: Inner Loop (N iterations)
```python
def _inner_step(carry, inputs):
    # Update agent parameters using current meta-network
    new_learner_state, new_actor_state, metrics = agent.learner_step(
        rng, rollout, learner_state, actor_state,
        update_rule_params, is_meta_training=True
    )
    # Update value function
    new_value_state, _, _ = agent.value_fn.update(
        value_state, rollout, logits
    )
    return (update_rule_params, new_learner_state,
            new_actor_state, new_value_state), metrics
```

Use `jax.lax.scan` to perform N inner steps, updating agent parameters each time.

#### Step 2: Validation Evaluation
```python
# Run inference on validation rollout with updated agent
agent_rollout_on_valid = agent.actor_step(
    actor_params=new_learner_state.params,
    rng=valid_rng,
    timestep=valid_rollout.to_env_timestep(),
    actor_state=valid_rollout.first_state()
)

# Calculate value estimates on validation rollout
value_out = agent.value_fn.get_value_outs(
    new_value_state, valid_rollout, agent_rollout_on_valid['logits']
)
```

#### Step 3: Meta-Loss Computation
```python
def _outer_loss(update_rule_params, agent_state, train_rollouts, valid_rollout, rng):
    # Perform inner steps
    (_, new_learner_state, new_actor_state, new_value_state), _ = jax.lax.scan(
        _inner_step, initial_state, (train_rollouts, learner_rngs)
    )

    # Evaluate on validation
    actions_on_valid = valid_rollout.actions[:-1]
    logits_on_valid = agent_rollout_on_valid['logits'][:-1]
    adv_t = value_out.normalized_adv

    # Policy gradient loss (main component)
    pg_loss = policy_gradient_loss(logits_on_valid, actions_on_valid, adv_t)

    # Regularization terms
    reg_loss = 0
    reg_loss += -1e-2 * entropy(logits_on_valid)  # Entropy bonus
    reg_loss += 1e-3 * (y_entropy_loss + z_entropy_loss)  # Prediction entropy
    reg_loss += 1e-3 * mean_squared_regularizers  # Mean regularizers
    reg_loss += 1e-2 * KL(target_policy || meta_policy)  # Target consistency

    # Total meta-loss
    meta_loss = pg_loss + reg_loss

    return meta_loss, (new_agent_state, train_metrics, meta_log)
```

**Key aspects:**
- **Policy gradient loss**: Main signal, measures how well the updated agent performs
- **Entropy regularizers**: Encourage exploration
- **Consistency regularizers**: Keep meta-network outputs consistent with target network
- **Stop gradient**: Advantages are stop-gradients to prevent meta-network from exploiting value function

#### Step 4: Compute Meta-Gradient
```python
meta_grads, outputs = jax.grad(_outer_loss, has_aux=True)(
    update_rule_params, agent_state, train_rollouts, valid_rollout, rng
)
```

The gradient flows through:
1. Validation policy gradient loss
2. All N inner agent updates
3. All meta-network applications during inner updates
4. Back to meta-network parameters η

This is **differentiating through the entire inner optimization loop**.

### 6. Meta-Update Step (meta_train.ipynb:cell-7)

The `meta_update()` function aggregates across multiple agents:

**Process:**

#### Step 1: Generate Rollouts for All Agents
```python
for agent_i in range(num_agents):
    # Generate num_inner_steps training rollouts
    for step_i in range(num_inner_steps):
        state, rollouts[step_i] = agent.unroll_actor(state, rng, rollout_len)
    train_rollouts[agent_i] = stack(rollouts)

    # Generate validation rollout (2x longer)
    agents_states[agent_i], valid_rollouts[agent_i] = agent.unroll_actor(
        state, rng, 2 * rollout_len
    )
```

Configuration (meta_train.ipynb:cell-8):
- `num_agents = 2`: Population size
- `rollout_len = 16`: Length of each rollout
- `num_inner_steps = 2`: Number of agent updates per meta-update
- `batch_size_per_device = 32`: Parallel environments

#### Step 2: Calculate Meta-Gradients for Each Agent
```python
for agent_i in range(num_agents):
    meta_grads[agent_i], (agents_states[agent_i], metrics, meta_log) = \
        calculate_meta_gradient(
            update_rule_params, agents_states[agent_i],
            train_rollouts[agent_i], valid_rollouts[agent_i],
            rng, agents[agent_i]
        )
```

Each agent independently:
1. Performs inner loop updates on its train rollouts
2. Evaluates on its validation rollout
3. Computes meta-gradient

#### Step 3: Aggregate and Update Meta-Parameters
```python
# Average meta-gradients across all agents
avg_meta_gradient = jax.tree.map(
    lambda x: x.mean(axis=0), tree_stack(meta_grads)
)

# Apply meta-optimizer (Adam)
meta_update, meta_opt_state = meta_opt.update(avg_meta_gradient, meta_opt_state)

# Update meta-network parameters
update_rule_params = optax.apply_updates(update_rule_params, meta_update)
```

**Meta-optimizer configuration:**
- Optimizer: Adam with learning rate 5e-4
- Gradient aggregation: Mean across agent population

### 7. Complete Training Loop (meta_train.ipynb:cell-9)

```python
for meta_step in range(num_steps):  # num_steps = 800
    # Replicate parameters across devices for parallel execution
    step_update_rule_params = jax.device_put_replicated(update_rule_params, devices)
    step_meta_opt_state = jax.device_put_replicated(meta_opt_state, devices)
    step_agents_states = jax.device_put_replicated(agents_states, devices)

    # Generate random seeds for each device
    step_rngs = jax.random.split(rng, len(devices))

    # Execute meta-update in parallel across devices
    (step_update_rule_params, step_meta_opt_state,
     step_agents_states, metrics, meta_log) = jitted_meta_update(
        update_rule_params=step_update_rule_params,
        meta_opt_state=step_meta_opt_state,
        agents_states=step_agents_states,
        rng=step_rngs,
    )

    # Collect metrics from devices
    metrics, meta_log = jax.device_get((metrics, meta_log))
```

**Parallelization:**
- Uses `jax.pmap` to parallelize across multiple devices (TPUs/GPUs)
- Each device runs the same meta-update with different random seeds
- Gradients are averaged across devices using `jax.lax.pmean`

## Agent Loss Components (disco.py:210-323)

The agent's loss has two parts:

### 1. Meta-Differentiable Losses (agent_loss)

These losses have gradients w.r.t. meta-network parameters when `backprop=True`:

```python
# Parse agent outputs (drop last timestep)
logits = agent_out['logits'][:-1]  # Policy logits
y = agent_out['y'][:-1]  # Observation-conditional predictions
z_a = agent_out['z'][:-1][actions]  # Action-conditional predictions

# Parse meta-network targets
pi_hat = meta_out['pi']  # Policy target
y_hat = meta_out['y']  # y target
z_hat = meta_out['z']  # z target

# KL divergence losses
pi_loss = KL(pi_hat || logits)  # Policy loss
y_loss = KL(y_hat || y)  # Observation prediction loss
z_loss = KL(z_hat || z_a)  # Action prediction loss

# Auxiliary policy prediction loss
aux_pi_a = agent_out['aux_pi'][:-1][actions]  # 1-step policy predictor
aux_target = agent_out['logits'][1:]  # Next-step policy
aux_policy_loss = KL(stop_grad(aux_target) || aux_pi_a)

# Total loss
total_loss = (pi_cost * pi_loss +
              y_cost * y_loss +
              z_cost * z_loss +
              aux_policy_cost * aux_policy_loss)
```

**Hyperparameters** (agent.py:319-378):
- `pi_cost = 1.0`: Policy loss weight
- `y_cost = 1.0`: y prediction loss weight
- `z_cost = 1.0`: z prediction loss weight
- `aux_policy_cost = 1.0`: Auxiliary policy loss weight

### 2. Non-Meta Losses (agent_loss_no_meta)

These losses have stop-gradient on targets to **not interfere with meta-gradient**:

```python
# Q-value loss
q_a = agent_out['q'][:-1][actions]  # Agent's Q-value for taken action
td = stop_grad(meta_out['q_td'])  # TD target from value function
value_loss = value_loss_from_td(q_a, td)

# Total non-meta loss
loss = value_cost * value_loss
```

**Hyperparameters:**
- `value_cost = 0.2`: Q-value loss weight

**Rationale:** The Q-value is used for computing advantages (which feed into meta-network), so we stop-gradient the Q-loss to avoid meta-network manipulating advantages.

## Value Function (value_fn.py:31-199)

A **separate value network** is used only during meta-training:

**Purpose:**
- Provides advantage estimates for meta-gradient computation
- Not part of the discovered algorithm (not used during evaluation)

**Architecture:**
- MLP with layers (256, 256)
- Learning rate: 1e-3
- TD-lambda: 0.96
- Discount: 0.99

**Update:**
```python
def update(self, value_state, rollout, target_logits):
    # Compute value estimates and advantages
    value_outs, net_out, adv_ema_state, td_ema_state = \
        self.get_value_outs(value_state, rollout, target_logits)

    # Compute TD loss
    value_loss = value_loss_from_td(net_out[:-1], stop_grad(value_outs.normalized_td))

    # Update value parameters
    grads = jax.grad(value_loss)(value_state.params)
    new_params = optax.apply_updates(value_state.params, optimizer.update(grads))

    return new_state, value_outs, log
```

The value function is updated during the inner loop alongside agent parameters.

## Key Design Choices

### 1. Why Learn Targets Instead of Loss Functions?

The meta-network generates **targets** (pi_hat, y_hat, z_hat) rather than directly computing losses. This is more expressive because:
- Targets can be based on bootstrapping (using future predictions)
- Targets can combine information in complex ways
- The agent uses simple KL divergence, making it stable and efficient

### 2. Reverse Unrolling for Bootstrapping

The per-trajectory LSTM unrolls **backwards** to enable bootstrapping:
- At timestep t, the LSTM has already seen timesteps t+1, t+2, ..., T
- This allows targets to incorporate information about future states
- Similar to how TD-lambda uses future rewards

### 3. Meta-LSTM for Lifetime Learning

The Meta-LSTM maintains state across rollouts:
- Processes aggregated statistics from each rollout
- Evolves throughout the agent's lifetime
- Enables curriculum learning and adaptive update rules

### 4. Population of Agents

Training uses a population of agents (default: 2):
- Increases diversity of training data
- Meta-gradients averaged across agents
- More robust meta-network

### 5. Exponential Moving Average of Agent Parameters

Target network uses EMA of agent parameters:
- Provides stable bootstrapping targets
- Coefficient: 0.9 (default)
- Updated after each agent update: `target = 0.9 * target + 0.1 * current`

## Computational Flow Summary

```
For each meta-step:
  For each agent in population:
    1. Collect train rollouts (num_inner_steps × rollout_len steps)
    2. Collect validation rollout (2 × rollout_len steps)

    3. Inner loop (repeated num_inner_steps times):
       a. Agent.learner_step():
          - Unroll agent network → get (logits, y, z, q)
          - Apply meta-network → get targets (pi_hat, y_hat, z_hat)
          - Compute loss = KL divergences + value loss
          - Gradient descent on agent parameters θ

       b. ValueFunction.update():
          - Compute advantages and TD targets
          - Gradient descent on value parameters

    4. Outer loop:
       a. Evaluate updated agent on validation rollout
       b. Compute meta-loss:
          - Policy gradient with advantages
          - Entropy regularizers
          - Consistency regularizers
       c. Compute meta-gradient ∂(meta-loss)/∂η
          (differentiates through all inner steps!)

  5. Aggregate meta-gradients across agents
  6. Update meta-network parameters η using Adam
```

## Initialization

### Agent Parameters
```python
# Random initialization for agent network
agent_params = agent.initial_learner_state(rng)
# Includes: network params, optimizer state, meta-state
```

### Meta-Network Parameters

Two options demonstrated in the notebook:

1. **Random initialization:**
```python
update_rule_params, _ = agent.update_rule.init_params(rng)
```

2. **Pre-trained Disco103:**
```python
# Load pre-trained weights from Google's release
update_rule_params = disco_103_params
```

The notebook shows meta-training from random initialization, but starting from Disco103 enables fine-tuning for new domains.

## Training Configuration

From meta_train.ipynb:cell-8:

```python
# Meta-training hyperparameters
num_steps = 800  # Meta-gradient steps
num_agents = 2  # Population size
rollout_len = 16  # Trajectory length
num_inner_steps = 2  # Agent updates per meta-update
batch_size_per_device = 32  # Parallel environments
meta_learning_rate = 5e-4  # Adam learning rate for meta-params

# Agent hyperparameters
agent_learning_rate = 5e-4
max_abs_update = 1.0

# Value function hyperparameters
value_learning_rate = 1e-3
value_discount = 0.99
value_td_lambda = 0.96

# Environment
env = CatchJittableEnvironment(
    batch_size=32,
    env_settings=dict(rows=5, columns=5)
)
```

## Metrics and Logging

The training loop tracks:

**Meta-level metrics:**
- `meta_loss`: Total meta-loss (policy gradient + regularizers)
- `pg_loss`: Policy gradient component
- `reg_loss`: Regularization component
- `meta_grad_norm`: Norm of meta-gradients
- `meta_up_norm`: Norm of meta-updates
- `rewards`: Average reward across agents
- `pos_rewards`: Count of positive rewards
- `neg_rewards`: Count of negative rewards

**Agent-level metrics:**
- `total_loss`: Agent's total loss
- `pi_loss`: Policy KL loss
- `aux_kl_loss`: Auxiliary policy loss
- `q_loss`: Q-value loss
- `entropy`: Policy entropy
- `global_gradient_norm`: Norm of agent gradients
- `global_update_norm`: Norm of agent updates

**Value function metrics:**
- `value_loss`: Value function TD loss
- `adv`: Advantages
- `normalized_adv`: Normalized advantages
- `value`: Value estimates

## Differences from Standard RL

| Aspect | Standard RL | DiscoRL Meta-Training |
|--------|-------------|----------------------|
| **What's being learned** | Agent policy/value | Update rule (meta-network) |
| **Optimization** | Single-level | Two-level (nested) |
| **Loss function** | Hand-designed (e.g., A3C) | Generated by meta-network |
| **Validation data** | Not typically used | Essential for meta-gradient |
| **Value function** | Part of algorithm | Only for meta-training |
| **Population** | Usually single agent | Multiple agents for diversity |
| **Backprop through** | Policy/value network | Entire optimization process |

## Key Insights

1. **Meta-learning discovers algorithmic components:** The meta-network learns to generate targets that lead to effective learning, discovering algorithmic principles.

2. **Differentiating through optimization:** The meta-gradient backpropagates through multiple steps of agent optimization, which is computationally expensive but powerful.

3. **Bootstrapping in meta-network:** The reverse LSTM enables using future information in targets, similar to TD learning.

4. **Separate concerns:** Agent loss is simple (KL divergence), complexity is in the meta-network.

5. **Generalization:** Once trained, the meta-network (Disco103) can be used as a drop-in replacement for hand-designed losses on new tasks.

## Files Reference

Key implementation files:
- `disco_rl/agent.py` (lines 248-316): Agent learner_step
- `disco_rl/update_rules/disco.py` (lines 112-323): DiscoUpdateRule
- `disco_rl/networks/meta_nets.py` (lines 45-227): Meta-network architecture
- `disco_rl/value_fns/value_fn.py` (lines 104-198): Value function for meta-training
- `disco_rl/colabs/meta_train.ipynb` (cells 5-9): Meta-training loop

## Conclusion

The meta-training process in `meta_train.ipynb` demonstrates a sophisticated meta-learning approach where:

1. **Agent learns to solve tasks** using targets from a meta-network (inner loop)
2. **Meta-network learns to generate good targets** by observing agent performance (outer loop)
3. **Two-level optimization** with gradients flowing through the entire inner optimization
4. **Population-based training** with multiple agents for robustness
5. **Rich meta-network architecture** with bootstrapping and lifetime learning

This approach discovered Disco103, which achieves state-of-the-art performance on Atari, ProcGen, and other benchmarks, demonstrating that meta-learned algorithms can outperform hand-designed ones.
