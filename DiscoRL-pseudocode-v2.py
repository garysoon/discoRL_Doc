"""
DiscoRL (Discovering Reinforcement Learning) Algorithm Pseudocode
=================================================================

Based on: "Discovering state-of-the-art reinforcement learning algorithms"
Nature, Vol 648, 11 December 2025
Authors: Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, et al.

This pseudocode outlines the complete DiscoRL algorithm including:
- Agent Network architecture and training
- Meta-Network architecture and training
- Discovery process (meta-optimization)
"""

# ==============================================================================
# PART 1: DATA STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================

@dataclass
class AgentOutputs:
    """Outputs produced by the Agent Network."""
    pi: Array  # Policy logits [B, A] - probability distribution over actions
    y: Array   # Observation-conditioned prediction [B, Y] - meta-learned semantics
    z: Array   # Action-conditioned prediction [B, A, Z] - meta-learned semantics
    q: Array   # Action-value function [B, A, num_bins] - pre-defined semantics
    p: Array   # Auxiliary policy prediction [B, A, A] - pre-defined semantics


@dataclass
class MetaNetworkOutputs:
    """Targets produced by the Meta-Network."""
    pi_hat: Array  # Policy target [T, B, A]
    y_hat: Array   # Observation-conditioned prediction target [T, B, Y]
    z_hat: Array   # Action-conditioned prediction target [T, B, Z]


@dataclass
class UpdateRuleInputs:
    """Inputs to the update rule / meta-network."""
    observations: Array      # [T+1, B, ...]
    actions: Array           # [T+1, B]
    rewards: Array           # [T, B]
    is_terminal: Array       # [T, B]
    agent_out: AgentOutputs  # [T+1, B, ...]
    behaviour_agent_out: AgentOutputs  # From behavior policy (for off-policy)
    target_agent_out: AgentOutputs     # From target network (EMA params)
    value_outputs: ValueOutputs        # Advantage estimates, TD errors, etc.


@dataclass
class LearnerState:
    """State maintained by the agent learner."""
    params: AgentParams           # Agent network parameters (theta)
    opt_state: OptimizerState     # Adam optimizer state
    meta_state: MetaState         # Meta-network RNN state + EMA states


@dataclass
class MetaState:
    """State maintained across agent updates for meta-network."""
    rnn_state: LSTMState          # Meta-RNN state (unrolled over agent updates)
    adv_ema_state: EmaState       # Exponential moving average for advantage normalization
    td_ema_state: EmaState        # Exponential moving average for TD normalization
    target_params: AgentParams    # Target network parameters (exponential moving average)


# ==============================================================================
# PART 2: AGENT NETWORK ARCHITECTURE
# ==============================================================================

class AgentNetwork:
    """
    Agent Network (parameterized by theta)

    Architecture:
    - Encoder: MLP or CNN torso to process observations
    - Flat outputs: policy (pi) and observation-conditioned prediction (y)
    - Action-conditional model (LSTM): produces z, q, and auxiliary policy (p)

    Reference: disco_rl/networks/nets.py:127-159, disco_rl/networks/action_models.py:36-105
    """

    def __init__(self, action_spec, prediction_size, num_value_bins):
        self.encoder = MLP(hidden_sizes=[512, 512])  # Torso network

        # Flat output heads (observation-conditioned)
        self.policy_head = Linear(num_actions)           # pi logits
        self.y_head = Linear(prediction_size)            # y prediction

        # Action-conditional model (LSTM-based, inspired by MuZero/Muesli)
        self.action_model = LSTMActionModel(
            lstm_size=128,
            head_mlp_hiddens=[128],
            out_spec={
                'z': prediction_size,      # Action-conditioned prediction
                'aux_pi': num_actions,     # Auxiliary policy prediction
                'q': num_value_bins,       # Categorical value function (601 bins)
            }
        )

    def forward(self, observation) -> AgentOutputs:
        """
        Forward pass through agent network.

        Reference: disco_rl/networks/nets.py:115-124
        """
        # Step 1: Encode observation
        embedding = self.encoder(flatten(observation))  # [B, H]

        # Step 2: Compute flat outputs (observation-conditioned)
        pi_logits = self.policy_head(embedding)  # [B, A]
        y = self.y_head(embedding)               # [B, Y]

        # Step 3: Compute action-conditional outputs via LSTM model
        root_state = self.action_model.root_embedding(embedding)  # LSTMState
        model_outputs = self.action_model.model_step(root_state)
        # model_outputs contains: z [B, A, Z], aux_pi [B, A, A], q [B, A, num_bins]

        return AgentOutputs(
            pi=pi_logits,
            y=y,
            z=model_outputs['z'],
            q=model_outputs['q'],
            p=model_outputs['aux_pi'],
        )

    def unroll(self, observations, should_reset) -> AgentOutputs:
        """
        Unroll network over a trajectory of observations.

        Reference: disco_rl/networks/nets.py:109-113
        """
        # Apply forward pass to each timestep
        return batch_apply(self.forward, observations)  # [T, B, ...]


class LSTMActionModel:
    """
    LSTM-based action-conditional model (inspired by MuZero/Muesli).

    Reference: disco_rl/networks/action_models.py:36-105
    """

    def __init__(self, lstm_size, head_mlp_hiddens, out_spec):
        self.lstm = LSTM(lstm_size)
        self.heads = {key: MLP(head_mlp_hiddens + [size])
                      for key, size in out_spec.items()}

    def root_embedding(self, state: Array) -> LSTMState:
        """Create root LSTM state from encoder output."""
        flat_state = flatten(state)
        cell = Linear(self.lstm_size)(flat_state)
        return LSTMState(hidden=tanh(cell), cell=cell)

    def model_step(self, embedding: LSTMState) -> dict:
        """
        Perform model transition for ALL actions simultaneously.
        Uses weight sharing across action dimension.
        """
        num_actions = self.num_actions
        batch_size = embedding.cell.shape[0]

        # Create one-hot actions for all actions
        one_hot_actions = eye(num_actions)  # [A, A]
        batched_actions = tile(one_hot_actions, [batch_size, 1])  # [B*A, A]

        # Repeat embedding for each action
        all_actions_embed = tree_map(
            lambda x: repeat(x, num_actions, axis=0),
            embedding
        )  # [B*A, H]

        # LSTM transition
        lstm_output, _ = self.lstm(batched_actions, all_actions_embed)  # [B*A, H]

        # Compute outputs via MLP heads
        outputs = {}
        for key, head in self.heads.items():
            pred = head(lstm_output)  # [B*A, output_size]
            outputs[key] = reshape(pred, [batch_size, num_actions, -1])

        return outputs


# ==============================================================================
# PART 3: META-NETWORK ARCHITECTURE
# ==============================================================================

class MetaNetwork:
    """
    Meta-Network (parameterized by eta)

    Produces targets (pi_hat, y_hat, z_hat) towards which agent updates its predictions.
    Uses backward LSTM to process trajectory for n-step bootstrapping.

    Architecture:
    - Input embedding networks (shared across actions)
    - Backward LSTM (processes trajectory in reverse for bootstrapping)
    - Meta-RNN (processes across agent updates for lifetime adaptation)
    - Output decoders for targets

    Reference: disco_rl/networks/meta_nets.py:45-158
    """

    def __init__(self, hidden_size=256, prediction_size=600, embedding_size=(16, 1)):
        # Input embedding networks
        self.y_net = MLP(embedding_size)           # For y predictions
        self.z_net = MLP(embedding_size)           # For z predictions
        self.policy_net = Conv1DNet(channels=[16, 2])  # For policy inputs

        # Core LSTM (unrolled backward for bootstrapping)
        self.per_trajectory_lstm = LSTM(hidden_size)

        # Meta-RNN (unrolled forward across agent updates)
        self.meta_rnn = MetaLSTM(hidden_size=128)

        # Output heads
        self.y_output = Linear(prediction_size)
        self.z_output = Linear(prediction_size)
        self.policy_target_net = Conv1DNet(channels=[16])
        self.policy_output = Linear(1)

    def forward(self, inputs: UpdateRuleInputs, meta_rnn_state: LSTMState) -> MetaNetworkOutputs:
        """
        Forward pass of meta-network.

        Reference: disco_rl/networks/meta_nets.py:78-157
        """
        T, B, num_actions = inputs.agent_out.pi.shape

        # Step 1: Construct input embeddings
        # Process various inputs (policy, predictions, rewards, advantages, etc.)
        x, policy_emb = self._construct_input(inputs)  # [T, B, E], [T, B, A, C]

        # Step 2: Unroll backward LSTM for bootstrapping
        # This allows targets to depend on future predictions (n-step returns)
        should_reset_bwd = inputs.should_reset_mask_bwd[:-1]  # [T, B]
        x, _ = dynamic_unroll(
            self.per_trajectory_lstm,
            (x, should_reset_bwd),
            initial_state=zeros([B, hidden_size]),
            reverse=True  # BACKWARD unrolling for bootstrapping
        )  # [T, B, H]

        # Step 3: Multiplicative interaction with meta-RNN output
        # Allows adaptation based on agent's learning history
        meta_rnn_output = self.meta_rnn.output(meta_rnn_state)  # [H]
        x = x * Linear(hidden_size)(meta_rnn_output)  # [T, B, H]

        # Step 4: Compute y_hat and z_hat targets
        y_hat = batch_apply(self.y_output, x)  # [T, B, Y]
        z_hat = batch_apply(self.z_output, x)  # [T, B, Z]

        # Step 5: Compute policy target (pi_hat)
        # Combine LSTM output with per-action embeddings
        w = repeat(expand_dims(x, 2), num_actions, axis=2)  # [T, B, A, H]
        w = concatenate([w, policy_emb], axis=-1)           # [T, B, A, H+C]
        w = self.policy_target_net(w)                       # [T, B, A, O]
        pi_hat = squeeze(batch_apply(self.policy_output, w), -1)  # [T, B, A]

        # Step 6: Update meta-RNN state
        meta_input_emb = batch_apply(Linear(1), x)  # [T, B, 1]
        new_meta_rnn_state = self.meta_rnn.unroll(inputs,
                                                   {'y': y_hat, 'meta_input_emb': meta_input_emb},
                                                   meta_rnn_state)

        return MetaNetworkOutputs(pi_hat=pi_hat, y_hat=y_hat, z_hat=z_hat), new_meta_rnn_state

    def _construct_input(self, inputs: UpdateRuleInputs) -> tuple[Array, Array]:
        """
        Construct meta-network inputs from agent outputs and environment signals.

        Inputs include (Reference: disco_rl/update_rules/disco.py:326-428):
        - Policy (current, behavior, target) with softmax and action selection
        - Predictions y, z (current and target) with softmax
        - Rewards (sign-log transformed)
        - Episode termination indicators
        - Value estimates and advantages (sign-log transformed)
        - Normalized advantages

        Reference: disco_rl/networks/meta_nets.py:256-331
        """
        inputs_list = []

        # Policy-related inputs
        policy = softmax(inputs.agent_out.pi)  # [T+1, B, A]
        policy_a = select_by_action(policy[:-1], inputs.actions[:-1])  # [T, B]
        inputs_list.append(policy_a)

        # Behavior policy
        behaviour_policy_a = select_by_action(
            softmax(inputs.behaviour_agent_out.pi)[:-1],
            inputs.actions[:-1]
        )
        inputs_list.append(behaviour_policy_a)

        # Target policy
        target_policy_a = select_by_action(
            softmax(inputs.target_agent_out.pi)[:-1],
            inputs.actions[:-1]
        )
        inputs_list.append(target_policy_a)

        # Rewards (sign-log transform for scale invariance)
        inputs_list.append(signed_log1p(inputs.rewards))  # [T, B]

        # Episode termination (as discounts)
        inputs_list.append(1.0 - inputs.is_terminal)  # [T, B]

        # Value and advantage estimates
        inputs_list.append(signed_log1p(td_pair(inputs.value_outputs.value)))
        inputs_list.append(signed_log1p(inputs.value_outputs.adv))
        inputs_list.append(inputs.value_outputs.normalized_adv)

        # Prediction embeddings (y)
        y_emb = self.y_net(softmax(inputs.agent_out.y))
        y_emb_td = td_pair(y_emb)  # Concat t and t+1 for TD-like computation
        inputs_list.append(y_emb_td)

        # Target y embeddings
        target_y_emb = self.y_net(softmax(inputs.target_agent_out.y))
        target_y_emb_td = td_pair(target_y_emb)
        inputs_list.append(target_y_emb_td)

        # z embeddings (action-conditioned)
        z_emb = self.z_net(softmax(inputs.agent_out.z))  # [T+1, B, A, E]
        z_emb_a = select_by_action(z_emb[:-1], inputs.actions[:-1])  # Selected action
        z_emb_avg = pi_weighted_avg(z_emb, policy)  # Policy-weighted average
        z_emb_max = max(z_emb, axis=2)  # Max over actions
        inputs_list.extend([z_emb_a, td_pair(z_emb_avg), td_pair(z_emb_max)])

        # Concatenate all inputs
        x = concatenate(inputs_list, axis=-1)  # [T, B, total_dim]

        # Action-conditional embeddings for policy target
        policy_emb = self.policy_net(...)  # [T, B, A, C]

        return x, policy_emb


class MetaLSTM:
    """
    Meta-RNN that processes across agent updates (not timesteps).
    Allows meta-network to adapt based on learning dynamics.

    Reference: disco_rl/networks/meta_nets.py:160-227
    """

    def __init__(self, hidden_size=128):
        self.lstm = LSTM(hidden_size)
        self.embedding_net = MLP([16])

    def unroll(self, inputs, meta_out, state: LSTMState) -> LSTMState:
        """
        Update meta-RNN state given a batch of trajectories.
        Called once per agent update (not per timestep).
        """
        # Embed trajectory information
        x = self._embed_trajectory(inputs, meta_out)  # [T, B, E]

        # Average over batch and time dimensions
        x_avg = mean(x, axis=(0, 1))  # [E]

        # If running under pmap, average across devices
        x_avg = pmean(x_avg, axis_name)

        # Single LSTM step
        _, new_state = self.lstm(x_avg, state)
        return new_state

    def initial_state(self) -> LSTMState:
        return self.lstm.initial_state(batch_size=None)

    def output(self, state: LSTMState) -> Array:
        return state.hidden


# ==============================================================================
# PART 4: AGENT TRAINING (INNER LOOP)
# ==============================================================================

def agent_loss(agent_params, rollout, meta_out, hyper_params, backprop=False):
    """
    Compute agent loss: distance from predictions to meta-network targets.

    L(theta) = D(pi, pi_hat) + D(y, y_hat) + D(z_a, z_hat) + L_aux

    where D is KL-divergence and L_aux includes value and auxiliary policy losses.

    Reference: disco_rl/update_rules/disco.py:210-294
    """
    # Unroll agent network on trajectory
    agent_out = agent_network.unroll(agent_params, rollout.observations)

    # Extract predictions (drop last timestep for alignment)
    logits = agent_out.pi[:-1]      # [T, B, A]
    y = agent_out.y[:-1]            # [T, B, Y]
    z_a = batch_lookup(agent_out.z[:-1], rollout.actions[:-1])  # [T, B, Z]

    # Extract targets from meta-network (stop gradient if not meta-training)
    pi_hat = meta_out.pi_hat        # [T, B, A]
    y_hat = meta_out.y_hat          # [T, B, Y]
    z_hat = meta_out.z_hat          # [T, B, Z]

    if not backprop:
        pi_hat, y_hat, z_hat = stop_gradient((pi_hat, y_hat, z_hat))

    # Compute KL-divergence losses (using softmax normalization)
    # KL(target || prediction) where both are treated as categorical distributions
    pi_loss = categorical_kl_divergence(pi_hat, logits)      # [T, B]
    y_loss = categorical_kl_divergence(y_hat, y)             # [T, B]
    z_loss = categorical_kl_divergence(z_hat, z_a)           # [T, B]

    # Auxiliary policy prediction loss (predict next step's policy)
    aux_pi = agent_out.p[:-1]                                # [T, B, A, A]
    aux_pi_a = batch_lookup(aux_pi, rollout.actions[:-1])    # [T, B, A]
    aux_target = stop_gradient(agent_out.pi[1:])             # [T, B, A]
    aux_loss = categorical_kl_divergence(aux_target, aux_pi_a)
    aux_loss = aux_loss * (1.0 - rollout.is_terminal)        # Mask terminal states

    # Value function loss (categorical cross-entropy with two-hot encoding)
    q_a = batch_lookup(agent_out.q[:-1], rollout.actions[:-1])  # [T, B, num_bins]
    q_target = meta_out.q_target                                # [T, B]
    value_loss = value_loss_from_td(q_a, meta_out.q_td)

    # Total loss
    total_loss = (
        hyper_params['pi_cost'] * pi_loss +
        hyper_params['y_cost'] * y_loss +
        hyper_params['z_cost'] * z_loss +
        hyper_params['aux_policy_cost'] * aux_loss +
        hyper_params['value_cost'] * value_loss
    )

    return mean(total_loss)


def agent_update_step(learner_state, rollout, meta_params, hyper_params, is_meta_training):
    """
    Single agent update step.

    Reference: disco_rl/agent.py:248-316
    """
    # Step 1: Unroll agent to get current predictions
    agent_out = agent_network.unroll(learner_state.params, rollout.observations)

    # Step 2: Compute target outputs using target network (EMA params)
    target_out = agent_network.unroll(
        learner_state.meta_state.target_params,
        rollout.observations
    )

    # Step 3: Compute value estimates (V-trace or Retrace)
    value_outs = compute_value_estimates(
        agent_out, target_out, rollout,
        discount=hyper_params['discount_factor'],
        lambda_=hyper_params['value_fn_td_lambda']
    )

    # Step 4: Prepare inputs for meta-network
    update_rule_inputs = UpdateRuleInputs(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards,
        is_terminal=rollout.is_terminal,
        agent_out=agent_out,
        behaviour_agent_out=rollout.behaviour_agent_out,
        target_agent_out=target_out,
        value_outputs=value_outs,
    )

    # Step 5: Apply meta-network to get targets
    meta_out, new_meta_rnn_state = meta_network.forward(
        meta_params,
        learner_state.meta_state.rnn_state,
        update_rule_inputs
    )

    # Step 6: Compute gradient of agent loss
    grad_fn = grad(agent_loss, has_aux=True)
    grads, aux = grad_fn(
        learner_state.params,
        rollout=rollout,
        meta_out=meta_out,
        hyper_params=hyper_params,
        backprop=is_meta_training
    )

    # Step 7: Average gradients across devices (if distributed)
    grads = pmean(grads, axis_name='batch')

    # Step 8: Apply optimizer update (Adam)
    updates, new_opt_state = optimizer.update(
        grads,
        learner_state.opt_state,
        learner_state.params
    )
    new_params = apply_updates(learner_state.params, updates)

    # Step 9: Update target parameters (exponential moving average)
    coeff = hyper_params['target_params_coeff']  # e.g., 0.9
    new_target_params = tree_map(
        lambda old, new: old * coeff + (1 - coeff) * new,
        learner_state.meta_state.target_params,
        new_params
    )

    # Step 10: Update meta state
    new_meta_state = MetaState(
        rnn_state=new_meta_rnn_state,
        adv_ema_state=value_outs.adv_ema_state,
        td_ema_state=value_outs.td_ema_state,
        target_params=new_target_params,
    )

    return LearnerState(
        params=new_params,
        opt_state=new_opt_state,
        meta_state=new_meta_state
    )


# ==============================================================================
# PART 5: META-OPTIMIZATION (OUTER LOOP) - DISCOVERY PROCESS
# ==============================================================================

def meta_gradient(meta_params, agents, environments, num_inner_steps=20):
    """
    Compute meta-gradient for updating meta-network parameters.

    Meta-objective: J(eta) = E[J(theta)] where theta evolves according to RL rule
    Meta-gradient: nabla_eta J(eta) = nabla_eta theta * nabla_theta J(theta)

    Uses backpropagation through the agent update process.

    Reference: Paper Section "Meta-optimization" (page 314)
    """
    total_meta_grad = zeros_like(meta_params)

    for agent, env in zip(agents, environments):
        # Initialize agent
        learner_state = agent.initial_learner_state()

        # Collect trajectory of agent parameters through updates
        param_trajectory = [learner_state.params]

        # Inner loop: update agent N times (with gradient tracking)
        for step in range(num_inner_steps):
            # Collect rollout from environment
            rollout = collect_rollout(agent, env, learner_state)

            # Update agent (keeping computation graph for meta-gradient)
            learner_state = agent_update_step(
                learner_state,
                rollout,
                meta_params,
                hyper_params,
                is_meta_training=True  # Keep gradients through targets
            )
            param_trajectory.append(learner_state.params)

        # Compute meta-objective: expected return of final agent
        # Using advantage actor-critic with meta-value function
        final_rollout = collect_rollout(agent, env, learner_state)
        meta_objective = compute_meta_objective(
            learner_state.params,
            final_rollout,
            meta_value_function
        )

        # Backpropagate through entire update trajectory
        # nabla_eta J = nabla_eta theta_N * nabla_theta_N J
        agent_meta_grad = grad(meta_objective, meta_params)

        # Accumulate gradients
        total_meta_grad = tree_map(add, total_meta_grad, agent_meta_grad)

    # Average over all agents
    return tree_map(lambda x: x / len(agents), total_meta_grad)


def compute_meta_objective(agent_params, rollout, meta_value_fn):
    """
    Meta-objective: maximize expected discounted return.

    J(theta) = E[sum_t gamma^t * r_t]

    Estimated using advantage actor-critic with a meta-value function.

    Reference: Paper Section "Meta-optimization" (page 314)
    """
    # Compute value estimates using meta-value function
    values = meta_value_fn(rollout.observations)

    # Compute advantages (V-trace for off-policy correction)
    advantages = vtrace_advantages(
        values, rollout.rewards, rollout.discounts, rollout.rho
    )

    # Normalize advantages for stability
    advantages = (advantages - mean(advantages)) / (std(advantages) + eps)

    # Policy gradient loss
    log_probs = log_softmax(agent_params, rollout.observations)
    log_pi_a = batch_lookup(log_probs, rollout.actions)
    policy_loss = -mean(log_pi_a * stop_gradient(advantages))

    return policy_loss


def meta_training_loop(num_meta_steps, num_environments=128):
    """
    Main discovery loop for DiscoRL.

    Reference: Paper Section "Implementation details" (page 319)
    """
    # Initialize meta-network parameters
    meta_params = meta_network.init_params()
    meta_optimizer = Adam(learning_rate=0.001, gradient_clip=1.0)
    meta_opt_state = meta_optimizer.init(meta_params)

    # Initialize population of agents and environments
    agents = [Agent() for _ in range(num_environments)]
    environments = sample_environments(num_environments)  # From Atari, ProcGen, etc.

    # Initialize meta-value function (for computing meta-gradient)
    meta_value_fn = ValueFunction()

    for meta_step in range(num_meta_steps):
        # Reset agents periodically to encourage fast learning
        if should_reset_agents(meta_step):
            agents = reset_agents(agents)

        # Compute meta-gradient from population
        meta_grad = meta_gradient(
            meta_params, agents, environments,
            num_inner_steps=20  # Sliding window for tractability
        )

        # Apply per-agent Adam normalization before averaging
        # (helps balance gradient magnitudes across environments)
        meta_grad = normalize_per_agent_gradients(meta_grad)

        # Meta-regularization losses
        entropy_reg = compute_prediction_entropy_regularization(agents)
        kl_reg = compute_policy_kl_regularization(agents)

        # Update meta-parameters
        meta_updates, meta_opt_state = meta_optimizer.update(
            meta_grad + entropy_reg + kl_reg,
            meta_opt_state,
            meta_params
        )
        meta_params = apply_updates(meta_params, meta_updates)

        # Update meta-value function
        meta_value_fn.update(agents)

        # Logging
        log_metrics(meta_step, agents, meta_params)

    return meta_params  # The discovered RL rule (DiscoRL)


# ==============================================================================
# PART 6: VALUE FUNCTION UTILITIES
# ==============================================================================

def compute_value_estimates(agent_out, target_out, rollout, discount, lambda_):
    """
    Compute value estimates using Retrace (for Q-values) or V-trace (for V-values).

    Reference: disco_rl/value_fns/value_utils.py:35-250
    """
    # Extract Q-values from agent output
    q_net_out = agent_out.q           # [T+1, B, A, num_bins]
    target_q_net_out = target_out.q   # [T+1, B, A, num_bins]

    # Convert categorical Q-values to scalar
    q_values = value_logits_to_scalar(q_net_out, max_abs_value=300.0)
    target_q_values = value_logits_to_scalar(target_q_net_out, max_abs_value=300.0)

    # Apply inverse transform (signed hyperbolic for scale invariance)
    q_values = signed_hyperbolic_inverse(q_values)
    target_q_values = signed_hyperbolic_inverse(target_q_values)

    # Compute state values as policy-weighted Q-values
    policy = softmax(agent_out.pi)
    values = sum(policy * q_values, axis=2)
    target_values = sum(policy * target_q_values, axis=2)

    # Importance weights for off-policy correction
    rho = importance_weight(agent_out.pi[:-1], rollout.logits[:-1], rollout.actions[:-1])

    # Compute Retrace targets for Q-values
    q_a = batch_lookup(target_q_values[:-1], rollout.actions[:-1])
    q_target = retrace(
        q_a, target_values, rollout.rewards,
        rollout.discounts * discount,
        lambda_ * clip(rho, max=1.0)
    )

    # Compute TD errors
    q_td = q_target - batch_lookup(q_values[:-1], rollout.actions[:-1])

    # Compute advantages
    adv = q_target - target_values[:-1]
    qv_adv = target_q_values - expand_dims(target_values, 2)

    return ValueOutputs(
        value=values,
        target_value=target_values,
        q_target=q_target,
        q_td=q_td,
        adv=adv,
        qv_adv=qv_adv,
        rho=rho,
    )


def value_loss_from_td(value_net_out, td, max_abs_value=300.0):
    """
    Compute value loss from TD error using categorical cross-entropy.

    Reference: disco_rl/value_fns/value_utils.py:582-617
    """
    # Get current value predictions
    values = value_logits_to_scalar(value_net_out, max_abs_value)
    values = signed_hyperbolic_inverse(values)

    # Construct target from TD
    value_target = stop_gradient(values + td)

    # Apply forward transform
    value_target = signed_hyperbolic(value_target)

    # Convert to two-hot categorical target
    value_target_probs = scalar_to_two_hot(value_target, num_bins=601, max_abs_value=300.0)

    # Categorical cross-entropy loss
    loss = softmax_cross_entropy(value_net_out, value_target_probs)

    return loss


# ==============================================================================
# PART 7: EVALUATION (USING DISCOVERED RULE)
# ==============================================================================

def evaluate_disco_rl(meta_params, environment, num_steps):
    """
    Evaluate the discovered RL rule (DiscoRL) on a new environment.

    The discovered rule generalizes to:
    - Unseen environments (different observation/action spaces)
    - Different agent architectures (larger networks)
    - Different hyperparameters (replay ratio, learning rate)
    """
    # Initialize agent with discovered rule
    agent = Agent(meta_params=meta_params)
    learner_state = agent.initial_learner_state()

    total_return = 0

    for step in range(num_steps):
        # Collect rollout
        rollout = collect_rollout(agent, environment, learner_state)
        total_return += sum(rollout.rewards)

        # Update agent using discovered rule
        learner_state = agent_update_step(
            learner_state,
            rollout,
            meta_params,
            hyper_params,
            is_meta_training=False  # Inference mode
        )

    return total_return


# ==============================================================================
# PART 8: MAIN ALGORITHM SUMMARY
# ==============================================================================

"""
DISCORL ALGORITHM SUMMARY
=========================

DISCOVERY PHASE (Meta-training):
--------------------------------
1. Initialize meta-network parameters (eta)
2. For each meta-step:
   a. Sample batch of environments from training set
   b. Initialize/reset population of agents
   c. For each agent update (inner loop):
      - Collect trajectory from environment
      - Apply meta-network to produce targets (pi_hat, y_hat, z_hat)
      - Update agent towards targets using KL-divergence loss
   d. Compute meta-gradient by backpropagating through agent updates
   e. Update meta-parameters using Adam

EVALUATION PHASE (Using discovered rule):
-----------------------------------------
1. Initialize new agent with discovered meta-network
2. For each episode:
   a. Collect trajectory from environment
   b. Apply meta-network to produce targets
   c. Update agent towards targets
   d. Repeat until convergence

KEY INNOVATIONS:
----------------
1. Meta-learning TARGETS (not loss function) - more expressive, includes semi-gradients
2. Learned prediction semantics (y, z) - not pre-defined like value functions
3. Backward LSTM for bootstrapping - enables n-step returns
4. Meta-RNN across updates - adapts to learning dynamics
5. Large-scale discovery from complex environments - Atari, ProcGen, DMLab

HYPERPARAMETERS (Disco-103):
----------------------------
- Prediction size: 600
- Meta-network hidden size: 256
- Meta-RNN hidden size: 128
- Value function bins: 601
- Max absolute value: 300.0
- Discount factor: 0.997
- TD-lambda: 0.95
- Target params EMA: 0.9
- Learning rate: 0.0003
- Meta learning rate: 0.001
"""
