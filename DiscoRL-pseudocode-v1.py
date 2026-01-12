"""
DiscoRL Algorithm Pseudocode
============================
Discovering Reinforcement Learning Algorithms Through Meta-Learning

This pseudocode outlines the DiscoRL algorithm which uses meta-learning to
autonomously discover reinforcement learning update rules.

Reference: Oh et al., "Discovering Reinforcement Learning Algorithms"
           Nature, 2020
"""

# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

@dataclass
class AgentNetworkOutputs:
    """Outputs from the Agent Network"""
    pi: Distribution      # Policy distribution π(a|s)
    y: Distribution       # Observation-conditioned prediction y(s)
    z: Distribution       # Action-conditioned prediction z(s,a)
    q: Distribution       # Q-value distribution q(s,a)
    p: Distribution       # Auxiliary policy prediction p(s,a)

@dataclass
class MetaNetworkTargets:
    """Targets produced by Meta-Network"""
    pi_hat: Distribution  # Policy target π̂
    y_hat: Distribution   # Observation-conditioned target ŷ
    z_hat: Distribution   # Action-conditioned target ẑ

@dataclass
class LearnerState:
    """State maintained by the Agent during learning"""
    params: AgentParams           # Agent network parameters θ
    opt_state: OptimizerState     # Optimizer state (e.g., Adam moments)
    meta_state: MetaRNNState      # Meta-RNN hidden state (per-lifetime)

@dataclass
class Rollout:
    """Experience collected from environment interaction"""
    observations: Array[T, B, obs_dim]     # Observations s_t
    actions: Array[T, B, action_dim]       # Actions a_t
    rewards: Array[T, B]                   # Rewards r_t
    discounts: Array[T, B]                 # Discount factors γ_t
    behavior_logits: Array[T, B, A]        # Behavior policy logits
    agent_outputs: AgentNetworkOutputs     # Agent predictions at each step


# =============================================================================
# PART 2: AGENT NETWORK ARCHITECTURE
# =============================================================================

class AgentNetwork:
    """
    Multi-Layer Perceptron (MLP) Agent Network

    Architecture:
        - Input: Observation embedding
        - Hidden layers: Multiple fully-connected layers with ReLU activation
        - Output heads: Separate heads for π, y, z, q, p

    Source: disco_rl/networks/nets.py
    """

    def __init__(self, config):
        self.num_hidden_layers = config.num_hidden_layers  # Default: 2
        self.hidden_size = config.hidden_size              # Default: 256
        self.num_actions = config.num_actions
        self.num_bins = config.num_bins                    # 601 bins for distributional outputs

    def forward(self, observation, action=None):
        """
        Forward pass through agent network

        Args:
            observation: State observation s_t (preprocessed)
            action: Optional action for action-conditioned outputs

        Returns:
            AgentNetworkOutputs containing π, y, z, q, p
        """
        # Embed observation
        h = observation

        # Pass through hidden layers
        for layer in range(self.num_hidden_layers):
            h = Linear(self.hidden_size)(h)
            h = ReLU(h)

        # Output heads
        pi_logits = Linear(self.num_actions)(h)           # Policy logits
        y_logits = Linear(self.num_bins)(h)               # Observation prediction

        # Action-conditioned outputs (computed for each action)
        z_logits = Linear(self.num_bins * self.num_actions)(h)  # Action prediction
        q_logits = Linear(self.num_bins * self.num_actions)(h)  # Q-value
        p_logits = Linear(self.num_actions * self.num_actions)(h)  # Aux policy

        return AgentNetworkOutputs(
            pi=Categorical(logits=pi_logits),
            y=Categorical(logits=y_logits),
            z=Categorical(logits=z_logits),
            q=Categorical(logits=q_logits),
            p=Categorical(logits=p_logits)
        )


# =============================================================================
# PART 3: META-NETWORK ARCHITECTURE
# =============================================================================

class MetaNetwork:
    """
    LSTM-based Meta-Network for generating update targets

    Architecture:
        - Input: Features from agent's experience (configurable)
        - Core: LSTM with action-invariant weight sharing
        - Output: Targets π̂, ŷ, ẑ for agent predictions

    Source: disco_rl/networks/meta_nets.py (lines 2261-2633)
    """

    def __init__(self, config):
        self.lstm_size = config.lstm_size        # Hidden size of LSTM
        self.num_actions = config.num_actions
        self.num_bins = config.num_bins

    def get_inputs(self, rollout, agent_outputs, option="full"):
        """
        Construct meta-network inputs from agent experience

        Input options (configurable search space):
            - y: observation-conditioned prediction
            - z: action-conditioned prediction
            - q: Q-value predictions
            - p: auxiliary policy predictions
            - pi: policy
            - r: reward (one-hot encoded)
            - gamma: discount
            - a: action (one-hot encoded)

        Returns:
            Concatenated input features for meta-network
        """
        inputs = []

        if 'y' in option:
            inputs.append(agent_outputs.y.probs)
        if 'z' in option:
            inputs.append(agent_outputs.z.probs)
        if 'q' in option:
            inputs.append(agent_outputs.q.probs)
        if 'p' in option:
            inputs.append(agent_outputs.p.probs)
        if 'pi' in option:
            inputs.append(agent_outputs.pi.probs)
        if 'r' in option:
            inputs.append(encode_reward_onehot(rollout.rewards))
        if 'gamma' in option:
            inputs.append(rollout.discounts)
        if 'a' in option:
            inputs.append(one_hot(rollout.actions, self.num_actions))

        return concatenate(inputs, axis=-1)

    def forward(self, inputs, lstm_state):
        """
        Forward pass through LSTM meta-network

        Args:
            inputs: Meta-network inputs [T, B, input_dim]
            lstm_state: Previous LSTM hidden state

        Returns:
            targets: MetaNetworkTargets
            new_lstm_state: Updated LSTM state
        """
        # Process through LSTM (action-invariant: shared across actions)
        lstm_output, new_lstm_state = LSTM(self.lstm_size)(inputs, lstm_state)

        # Generate targets for each prediction type
        pi_hat_logits = Linear(self.num_actions)(lstm_output)
        y_hat_logits = Linear(self.num_bins)(lstm_output)
        z_hat_logits = Linear(self.num_bins)(lstm_output)

        return MetaNetworkTargets(
            pi_hat=Categorical(logits=pi_hat_logits),
            y_hat=Categorical(logits=y_hat_logits),
            z_hat=Categorical(logits=z_hat_logits)
        ), new_lstm_state


class MetaRNN:
    """
    Per-Lifetime Meta-RNN (MetaLSTM)

    Unrolled across agent UPDATES (not timesteps) to track learning dynamics.
    This allows the meta-network to condition on the agent's learning history.

    Source: disco_rl/networks/meta_nets.py
    """

    def __init__(self, config):
        self.lstm_size = config.meta_rnn_size

    def forward(self, update_summary, meta_rnn_state):
        """
        Process agent update to update meta-RNN state

        Args:
            update_summary: Summary of agent's learning progress
            meta_rnn_state: Previous meta-RNN hidden state

        Returns:
            new_meta_rnn_state: Updated state for next agent update
        """
        output, new_state = LSTM(self.lstm_size)(update_summary, meta_rnn_state)
        return new_state


# =============================================================================
# PART 4: AGENT TRAINING (INNER LOOP)
# =============================================================================

def agent_loss(rollout, agent_outputs, meta_targets):
    """
    Compute agent loss as KL divergence to meta-network targets

    The agent is trained to match the targets produced by the meta-network.

    L_agent = D_KL(π̂ || π) + D_KL(ŷ || y) + D_KL(ẑ || z)

    Source: disco_rl/update_rules/disco.py (lines 4100-4200)

    Args:
        rollout: Experience data from environment
        agent_outputs: Current agent predictions
        meta_targets: Targets from meta-network

    Returns:
        Scalar loss value
    """
    # KL divergence for policy
    pi_loss = kl_divergence(meta_targets.pi_hat, agent_outputs.pi)

    # KL divergence for observation prediction
    y_loss = kl_divergence(meta_targets.y_hat, agent_outputs.y)

    # KL divergence for action prediction (only for taken action)
    z_loss = kl_divergence(meta_targets.z_hat, agent_outputs.z)

    # Total loss (averaged over time and batch)
    total_loss = mean(pi_loss + y_loss + z_loss)

    return total_loss


def agent_update_step(rollout, learner_state, meta_network, meta_params):
    """
    Single agent update step

    Source: disco_rl/agent.py (lines 1200-1350)

    Args:
        rollout: Experience data [T, B, ...]
        learner_state: Current agent state
        meta_network: Meta-network for generating targets
        meta_params: Meta-network parameters φ

    Returns:
        new_learner_state: Updated agent state
        logs: Training metrics
    """
    agent_params = learner_state.params
    meta_state = learner_state.meta_state

    # 1. Forward pass: get agent predictions
    agent_outputs = agent_network.forward(rollout.observations, rollout.actions)

    # 2. Construct meta-network inputs
    meta_inputs = meta_network.get_inputs(rollout, agent_outputs)

    # 3. Get targets from meta-network
    meta_targets, _ = meta_network.forward(meta_inputs, meta_state)

    # 4. Compute gradients of agent loss w.r.t. agent parameters
    def loss_fn(params):
        outputs = agent_network.forward(rollout.observations, params=params)
        return agent_loss(rollout, outputs, meta_targets)

    loss, grads = value_and_grad(loss_fn)(agent_params)

    # 5. Update agent parameters using optimizer (e.g., Adam)
    updates, new_opt_state = optimizer.update(grads, learner_state.opt_state)
    new_params = apply_updates(agent_params, updates)

    # 6. Update meta-RNN state (tracks learning dynamics)
    update_summary = compute_update_summary(agent_outputs, meta_targets)
    new_meta_state = meta_rnn.forward(update_summary, meta_state)

    new_learner_state = LearnerState(
        params=new_params,
        opt_state=new_opt_state,
        meta_state=new_meta_state
    )

    return new_learner_state, {"agent_loss": loss}


def train_agent_episode(env, learner_state, meta_params, num_updates):
    """
    Train agent for one episode (lifetime)

    Args:
        env: Environment to interact with
        learner_state: Initial agent state
        meta_params: Fixed meta-network parameters
        num_updates: Number of agent updates in lifetime

    Returns:
        final_state: Agent state after training
        total_return: Cumulative reward in episode
    """
    for update_idx in range(num_updates):
        # Collect rollout from environment
        rollout = collect_rollout(env, learner_state.params)

        # Perform agent update
        learner_state, logs = agent_update_step(
            rollout, learner_state, meta_network, meta_params
        )

    return learner_state, total_return


# =============================================================================
# PART 5: META-NETWORK TRAINING (OUTER LOOP)
# =============================================================================

def compute_advantages(rollout, value_estimates):
    """
    Compute advantage estimates using V-trace

    A_t = r_t + γ * V(s_{t+1}) - V(s_t) + γ * c_t * A_{t+1}

    Where c_t are truncated importance sampling ratios.

    Source: disco_rl/value_fns/value_utils.py (lines 2900-3100)

    Args:
        rollout: Experience data
        value_estimates: Value function predictions

    Returns:
        advantages: Advantage estimates for each timestep
    """
    T = rollout.length
    advantages = zeros([T])

    # Compute importance sampling ratios
    rhos = clip(exp(rollout.target_log_probs - rollout.behavior_log_probs),
                max=RHO_MAX)
    cs = clip(rhos, max=C_MAX)

    # Backward pass to compute advantages
    next_advantage = 0
    for t in reversed(range(T)):
        td_error = (rollout.rewards[t] +
                   rollout.discounts[t] * value_estimates[t+1] -
                   value_estimates[t])
        advantages[t] = rhos[t] * td_error + rollout.discounts[t] * cs[t] * next_advantage
        next_advantage = advantages[t]

    return advantages


def compute_q_targets_retrace(rollout, q_estimates):
    """
    Compute Q-value targets using Retrace algorithm

    Q^ret_t = r_t + γ * [c_t * (Q^ret_{t+1} - Q(s_{t+1}, a_{t+1})) + V(s_{t+1})]

    Source: disco_rl/value_fns/value_utils.py

    Args:
        rollout: Experience data
        q_estimates: Current Q-value estimates

    Returns:
        q_targets: Target Q-values for training
    """
    T = rollout.length
    q_targets = zeros([T])

    # Importance sampling ratios clipped for Retrace
    cs = clip(exp(rollout.target_log_probs - rollout.behavior_log_probs),
              max=RETRACE_C_MAX)

    # Backward pass
    next_q_target = q_estimates[-1]  # Bootstrap from final estimate
    for t in reversed(range(T)):
        v_next = compute_v_from_q(q_estimates[t+1], rollout.policy[t+1])
        q_targets[t] = rollout.rewards[t] + rollout.discounts[t] * (
            cs[t] * (next_q_target - q_estimates[t+1][rollout.actions[t+1]]) + v_next
        )
        next_q_target = q_targets[t]

    return q_targets


def meta_loss(meta_params, agent_trajectories, value_fn):
    """
    Compute meta-network loss using policy gradient

    The meta-network is trained to maximize agent's cumulative reward.
    Uses advantages for variance reduction.

    L_meta = -E[A_t * log π_meta(target | state)]

    Source: disco_rl/value_fns/value_fn.py (lines 3500-3600)

    Args:
        meta_params: Meta-network parameters φ
        agent_trajectories: Collection of agent training episodes
        value_fn: Value function for advantage estimation

    Returns:
        Scalar meta-loss value
    """
    total_loss = 0

    for trajectory in agent_trajectories:
        # Get value estimates
        value_estimates = value_fn.get_value(trajectory)

        # Compute advantages
        advantages = compute_advantages(trajectory, value_estimates)

        # Policy gradient loss with baseline
        meta_log_probs = compute_meta_log_probs(meta_params, trajectory)
        policy_loss = -mean(advantages * meta_log_probs)

        # Value function loss
        value_targets = compute_q_targets_retrace(trajectory, value_estimates)
        value_loss = mean((value_estimates - value_targets) ** 2)

        total_loss += policy_loss + VALUE_LOSS_COEF * value_loss

    return total_loss / len(agent_trajectories)


def meta_update_step(meta_params, value_fn_params, agent_trajectories):
    """
    Single meta-network update step

    Source: disco_rl/update_rules/disco.py

    Args:
        meta_params: Current meta-network parameters
        value_fn_params: Value function parameters
        agent_trajectories: Batch of agent training runs

    Returns:
        new_meta_params: Updated meta parameters
        new_value_fn_params: Updated value function parameters
        logs: Training metrics
    """
    # Compute gradients of meta-loss
    def loss_fn(params):
        return meta_loss(params, agent_trajectories, value_fn_params)

    loss, grads = value_and_grad(loss_fn)(meta_params)

    # Update meta-parameters using optimizer
    updates, new_opt_state = meta_optimizer.update(grads, meta_opt_state)
    new_meta_params = apply_updates(meta_params, updates)

    # Update value function
    new_value_fn_params = update_value_function(value_fn_params, agent_trajectories)

    return new_meta_params, new_value_fn_params, {"meta_loss": loss}


# =============================================================================
# PART 6: MAIN TRAINING LOOP
# =============================================================================

def disco_rl_training(config):
    """
    Main DiscoRL training loop

    Two-level optimization:
        - Inner loop: Agent learns using discovered update rule
        - Outer loop: Meta-network learns to improve the update rule

    Source: Overall algorithm from paper and disco_rl/agent.py
    """
    # Initialize networks
    agent_network = AgentNetwork(config)
    meta_network = MetaNetwork(config)
    meta_rnn = MetaRNN(config)
    value_fn = ValueFunction(config)

    # Initialize parameters
    meta_params = meta_network.init_params()
    value_fn_params = value_fn.init_params()
    meta_optimizer = Adam(config.meta_lr)

    # Meta-training loop
    for meta_iteration in range(config.num_meta_iterations):

        agent_trajectories = []

        # Collect multiple agent lifetimes
        for lifetime in range(config.num_lifetimes_per_meta_update):

            # Initialize fresh agent for each lifetime
            agent_params = agent_network.init_params()
            agent_optimizer = Adam(config.agent_lr)
            meta_rnn_state = meta_rnn.init_state()

            learner_state = LearnerState(
                params=agent_params,
                opt_state=agent_optimizer.init(agent_params),
                meta_state=meta_rnn_state
            )

            # Sample environment
            env = sample_training_environment(config.env_distribution)

            # Train agent for one lifetime (inner loop)
            trajectory = []
            for update_idx in range(config.num_agent_updates):
                # Collect experience
                rollout = collect_rollout(env, learner_state.params)

                # Store for meta-learning
                trajectory.append({
                    'rollout': rollout,
                    'agent_outputs': agent_network.forward(rollout.observations),
                    'learner_state': learner_state
                })

                # Agent update (inner loop)
                learner_state, _ = agent_update_step(
                    rollout, learner_state, meta_network, meta_params
                )

            agent_trajectories.append(trajectory)

        # Meta-network update (outer loop)
        meta_params, value_fn_params, logs = meta_update_step(
            meta_params, value_fn_params, agent_trajectories
        )

        # Logging
        print(f"Meta iteration {meta_iteration}: loss = {logs['meta_loss']}")

    return meta_params


# =============================================================================
# PART 7: VALUE FUNCTION FOR META-LEARNING
# =============================================================================

class ValueFunction:
    """
    Value function for meta-learning (baseline for variance reduction)

    Estimates expected future return from current agent state.
    Used to compute advantages for policy gradient.

    Source: disco_rl/value_fns/value_fn.py (lines 3418-3618)
    """

    def __init__(self, config):
        self.network = ValueNetwork(config)
        self.optimizer = Adam(config.value_lr)

    def get_value(self, trajectory):
        """
        Get value estimates for trajectory states

        Args:
            trajectory: Agent's training trajectory

        Returns:
            value_estimates: V(s) for each state in trajectory
        """
        return self.network.forward(trajectory.observations)

    def update(self, trajectories):
        """
        Update value function parameters

        Uses TD-lambda or Monte Carlo returns as targets.

        Args:
            trajectories: Collection of agent trajectories

        Returns:
            Updated value function parameters
        """
        # Compute value targets (e.g., discounted returns)
        targets = compute_value_targets(trajectories)

        # MSE loss between predictions and targets
        def loss_fn(params):
            predictions = self.network.forward(trajectories.observations, params)
            return mean((predictions - targets) ** 2)

        loss, grads = value_and_grad(loss_fn)(self.params)
        self.params = self.optimizer.update(self.params, grads)

        return self.params


# =============================================================================
# PART 8: OBSERVATION PREPROCESSING
# =============================================================================

def preprocess_observation(observation, config):
    """
    Transform raw environment observation to agent input

    For Atari:
        - Grayscale conversion
        - Resize to 84x84
        - Stack 4 frames
        - Normalize to [0, 1]

    For continuous control:
        - Normalize features
        - Optional: add noise for exploration

    Args:
        observation: Raw observation from environment
        config: Preprocessing configuration

    Returns:
        Processed observation suitable for agent network
    """
    if config.env_type == "atari":
        obs = rgb_to_grayscale(observation)
        obs = resize(obs, (84, 84))
        obs = obs / 255.0  # Normalize
        return obs
    else:
        return normalize(observation, config.obs_mean, config.obs_std)


# =============================================================================
# SUMMARY
# =============================================================================
"""
DiscoRL Key Components:

1. AGENT NETWORK (MLP):
   - Inputs: Preprocessed observations
   - Outputs: π (policy), y (obs prediction), z (action prediction), q (Q-value), p (aux policy)
   - Trained via KL divergence to meta-network targets

2. META-NETWORK (LSTM):
   - Inputs: Agent predictions, rewards, actions, discounts
   - Outputs: Targets π̂, ŷ, ẑ for agent to learn towards
   - Trained via policy gradient on agent's lifetime returns

3. META-RNN:
   - Tracks agent's learning dynamics across updates
   - Allows meta-network to condition on learning history

4. VALUE FUNCTION:
   - Estimates expected returns for variance reduction
   - Used to compute advantages for meta policy gradient

5. TWO-LEVEL OPTIMIZATION:
   - Inner loop: Agent learns in environment using discovered rule
   - Outer loop: Meta-network improves the update rule

6. KEY INNOVATIONS:
   - Bootstrapped predictions (y, z) for temporal credit assignment
   - Action-invariant meta-network for scalability
   - Meta-RNN for tracking learning progress
"""
