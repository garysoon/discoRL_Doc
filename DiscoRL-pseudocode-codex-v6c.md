# DiscoRL Algorithm Outline (v6c)

This is *pseudocode* that mirrors the structure of the minimal JAX harness in
`disco_rl/` (DeepMind's open-sourced reference implementation).


# Key components (as in the codebase)

- Agent network (policy + predictions):
  - Flat outputs: `logits` (policy), `y`
  - Action-conditional model outputs (per action): `z`, `aux_pi`, `q`
  (see `disco_rl/disco_rl/update_rules/disco.py` for specs, and
   `disco_rl/disco_rl/networks/` for the MLP + action-conditional LSTM model)

- Meta-network (update rule parameters, aka eta / learned rule):
  - Input: an `UpdateRuleInputs` rollout augmented with value/advantage features
  - Output targets: `pi_hat`, `y_hat`, `z_hat` (and `meta_input_emb`)
  (see `disco_rl/disco_rl/networks/meta_nets.py`)

- Value utilities:
  - Computes importance weights rho, (V-trace) advantages, (Retrace) Q targets,
    TD errors, and normalized versions via EMAs.
  (see `disco_rl/disco_rl/value_fns/value_utils.py`)

- Value function (meta-training only):
  - A separate learned V(s) baseline used to compute the *outer* meta-loss.
  (see `disco_rl/disco_rl/value_fns/value_fn.py` and `colabs/meta_train.ipynb`)


```python
# -----------------------------------------------------------------------------
# Data types (conceptual)
# -----------------------------------------------------------------------------


class ActorRollout:
  """
  Trajectory batch collected from environment interaction.

  Stores (time-major):
    observations[t, b, ...]
    actions[t, b]
    rewards[t, b]
    discounts[t, b]        # 0 on terminal transitions (or used as episode mask)
    logits[t, b, A]        # behavior policy logits (used for importance weights)
    agent_outs[t, b, ...]  # behavior policy outputs (includes y, z, q, aux_pi)
    states[t, b, ...]      # agent network state (empty for pure-MLP policy)
  """


class LearnerState:
  """
  Holds agent parameters and optimizer state, plus update-rule meta_state.

  params: agent network parameters
  opt_state: agent optimizer state (Adam-like, with clipping)
  meta_state:
    rnn_state: meta-network recurrent state (Meta-LSTM)
    adv_ema_state / td_ema_state: running stats for normalization
    target_params: EMA copy of agent params for bootstrapping (target policy)
  """


# -----------------------------------------------------------------------------
# Inner loop: Agent learning given a fixed update rule (meta params)
# -----------------------------------------------------------------------------


def actor_step(agent_params, actor_state, env_timestep, rng) -> tuple:
  """
  One environment interaction step.

  1) Forward agent network:
     agent_outs = AgentNet(obs)
       - logits: policy logits
       - y: auxiliary prediction logits
       - z[a]: action-conditional auxiliary prediction logits
       - aux_pi[a]: action-conditional 1-step policy-prediction logits
       - q[a]: action-conditional categorical value logits
  2) Sample action a ~ Softmax(logits)
  3) Step environment with a, record transition into rollout buffer
  """
  raise NotImplementedError


def learner_step(
    rollout: ActorRollout,
    learner_state: LearnerState,
    agent_net_state,
    update_rule_params,        # meta-network parameters (eta)
    is_meta_training: bool,    # controls stop_gradient on targets
) -> tuple:
  """
  One learner update on a rollout batch.

  A) Recompute agent outputs by unrolling the agent network on stored obs:
     agent_out = unroll_agent_net(learner_state.params, agent_net_state, rollout)

  B) Build UpdateRuleInputs:
     - behaviour_agent_out = rollout.agent_outs  (behavior policy outputs)
     - agent_out = agent_out                      (current policy outputs)
     - rewards, actions, is_terminal from rollout

  C) Compute meta targets via the meta-network (and value utilities):
     meta_out, new_meta_state = unroll_meta_net(
         meta_params=update_rule_params,
         params=learner_state.params,
         state=agent_net_state,
         meta_state=learner_state.meta_state,
         rollout=eta_inputs,
         hyper_params=...,            # pi_cost, y_cost, z_cost, etc.
     )
     where meta_out includes:
       pi_hat[t,b,A], y_hat[t,b,Y], z_hat[t,b,Y] (and value/adv features)

  D) Compute agent loss (per-step, then masked mean over non-terminal steps):
     - imitation-style losses vs meta targets:
         KL(pi_hat || logits) + KL(y_hat || y) + KL(z_hat || z_a)
       plus aux policy prediction loss:
         KL(stop_grad(logits_{t+1}) || aux_pi_a)
     - value loss (q_loss) computed separately to avoid meta-gradient interference

  E) Compute grads wrt agent params, apply optimizer, update learner_state.params.
  """
  raise NotImplementedError


def unroll_meta_net(
    meta_params,
    params,          # current agent params
    meta_state,
    rollout_inputs,  # UpdateRuleInputs
    hyper_params,
) -> tuple:
  """
  Computes targets and extra signals for the agent loss.

  1) Target policy outputs (bootstrapping network):
     target_out = unroll_agent_net(meta_state.target_params, rollout.observations)

  2) Value utilities (Retrace over Q; V-trace over V if needed):
     value_outs = get_value_outs(
         q_net_out=rollout.agent_out['q'],
         target_q_net_out=target_out['q'],
         pi_logits=rollout.agent_out['logits'],
         mu_logits=rollout.behaviour_agent_out['logits'],
         rewards, discounts, actions, ...
     )
     Produces: q_target, q_td, adv, normalized_adv, rho, etc.

  3) Augment rollout_inputs.extra_from_rule with value_outs + target_out.

  4) Meta-network forward:
     meta_out = MetaNet_LSTM(rollout_inputs_augmented, meta_state.rnn_state)
       returns:
         pi_hat, y_hat, z_hat, meta_input_emb

  5) Update meta_state:
     - meta_state.rnn_state <- new rnn state
     - adv/td EMA states <- updated stats
     - target_params <- EMA(target_params, params, coeff=target_params_coeff)
  """
  raise NotImplementedError


# -----------------------------------------------------------------------------
# Outer loop: Meta-training (learning update_rule_params)
# -----------------------------------------------------------------------------


def meta_training_loop(population_of_agents):
  """
  High-level meta-training loop (see `colabs/meta_train.ipynb`).

  For each meta step:
    For each agent in population:
      - Collect N training rollouts.
      - Run N inner learner steps with is_meta_training=True to adapt agent params.
      - Evaluate adapted agent on a validation rollout.
      - Compute outer loss on validation (policy-gradient with value_fn advantage)
        + regularizers (entropy, target-KL, etc.).
      - Compute gradient d(outer_loss)/d(update_rule_params).

    Aggregate meta-gradients across agents, apply meta-optimizer update:
      update_rule_params <- update_rule_params - alpha * meta_grad
  """
  raise NotImplementedError


# -----------------------------------------------------------------------------
# Evaluation / meta-evaluation
# -----------------------------------------------------------------------------


def meta_evaluate_fixed_update_rule(update_rule_params):
  """
  Evaluate a fixed discovered rule (e.g., Disco103 weights).

  Run standard RL training where:
    - update_rule_params is fixed (no meta updates)
    - is_meta_training=False, so targets are stop_grad'ed
    - agent trains via learner_step on rollouts (optionally with replay buffer)

  See `colabs/eval.ipynb` for the concrete training loop used in the repo.
  """
  raise NotImplementedError

```