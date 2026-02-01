# DiscoRL Notes (v6c)  

by CodeX gpt-5.2 xhigh, 2026-01-30  

---

# 笔记
- Categorical logits，指Neural network直接输出的vector，也就是raw outputs。其向量中每个元素 y_i 代表模型对选项i的信心的大小。y_i 取值(+♾️, -♾️)。它的缺点是无法作为概率使用，因此在AI算法中需要进一步处理。

- Normalized logits，是与 categorical logits 相对应的概念，最常用的处理方法是 Softmax()。 categorical logits经 softmax() 处理后，还是一个向量，但满足2个条件：向量中每个元素取值 [0, 1]，所有元素之和为1。它的好处是可以直接在AI算法中作为概率使用，能够直接作为许多后续环节的inputs。

- meta-network不从 env-observation 直接获取信息；而是从 agent-rollouts 间接的获取信息。  
  meta-network 与 env.observations 是隔离的。
  env, agent, meta-network 三者之间的 data-flow 是这样的：
  ```text
    env          --> observations --> 
    agent        --> (agent-rollouts, rewards, q_value, terminal_signals) --> 
    meta-network --> targets.

    从 env.observations 生成 agent.rollouts，其实分2步：  
    首先，agent根据observations生成predictions(π, y, z)；  
    然后，actor从 predictions(π, y, z)中选出要执行的 action 并在env中执行，此时才能获得 agent-rollouts及rewards等。  
  ```

- agent根据env.observations生成agent-rollouts时，不需要 meta-network 的 targets。但agent训练时，需要meta-targets用于计算Loss。  

- meta-network生成targets时，需要agent-rollouts。
  在meta-network训练时，需要agent生成的trajectories数据。 

- 问题：agent-rollouts 与 trajectory 的区别是什么 ？

- q_value 与 value 的区别？ 
  q_value 是对 (state, action) 估算出一个scalar值，计算方法是 Retrace path。例如 discoRL 中用的就是 q_value(s, a)。
  value 是对 (state) 估算出一个scalar值，计算方法是 V-trace path。例如AlphaZero中用的就是 value(s)。 
  当条件允许时，优先用 q_value(s, a)。
  先求出 q_value, 然后求出 advantage = q_value - value_baseline，最后用 q_advantage 参与计算 meta-gradient.

- value_targets 与 value_outs 的区别 ？
  下列回答有问题，可能是错的 ？？？
  value_targets 是 meta_network 这边的输出，与(π_hat, y_hat, z_hat)同属一组。
  value_outs 是 agent 这边的输出，与(π, y, z)同属一组。
  同时有了 value_targets 与 value_outs，用于计算 meta_gradients。
 

---

These notes are based on the local code in `disco_rl/` (the minimal JAX harness shipped with the DiscoRL paper), plus the two Colab notebooks under `disco_rl/colabs/`.

Notation used below:  
- `T`: unroll length (time), `B`: batch size, `A`: number of discrete actions.  
- For Disco103: `Y = prediction_size = 600`, categorical value bins `num_bins = 601`.  

---

## Q1. What is the neural network architecture of the Agent-network? Where is it defined? How is it defined?

### Answer
In this repo, the “Agent-network” is a Haiku policy network constructed by `nets.get_network(...)` and (for DiscoRL) configured as:

1) **Observation encoder (“torso”)**:   
an MLP that flattens the environment observation(s) and concatenates them,   
then applies an MLP with hidden sizes configured by `agent_settings.net_settings.net_args.dense` (e.g. `(512, 512)`).

2) **Flat heads** (computed from the torso embedding):   
linear/1-layer-MLP heads that produce:  
- `logits` (policy logits over actions, i.e. π after softmax).  
- `y` (a length-`Y` vector; used as categorical logits in DiscoRL).  

3) **Action-conditional “model head”** (MuZero/Muesli-inspired):   
from the same torso embedding, build a *root* LSTM state and do a single LSTM transition for **every possible action** (using one-hot action inputs),   
then decode per-action heads that produce:  
- `z` with shape `[B, A, Y]` (action-conditional vector prediction).  
- `aux_pi` with shape `[B, A, A]` (action-conditional next-policy prediction logits).  
- `q` with shape `[B, A, num_bins]` (action-conditional categorical value logits).  

So “Agent-network” = MLP torso + (flat heads) + (action-conditional LSTM model that produces per-action outputs).  

### Source code (with line numbers) + comments

**1) Agent creates the network using update-rule output specs**

`disco_rl/disco_rl/agent.py`:
```python
  90	    # Define the agent's neural network.
  91	    flat_out_spec = self.update_rule.flat_output_spec(self.single_action_spec)
  92	    model_out_spec = self.update_rule.model_output_spec(self.single_action_spec)
  93	    self._network = nets.get_network(
  94	        name=agent_settings.net_settings.name,
  95	        action_spec=self.single_action_spec,
  96	        out_spec=flat_out_spec,
  97	        model_out_spec=model_out_spec,
  98	        **agent_settings.net_settings.net_args,
  99	    )
```
Comments:  
- `flat_out_spec` determines which outputs come directly from the torso (`logits`, `y` for DiscoRL).  
- `model_out_spec` determines which outputs are produced per-action via the action-conditional model (`z`, `aux_pi`, `q` for DiscoRL).  

**2) The DiscoRL update rule defines which outputs the agent network must produce**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
   95	  def flat_output_spec(
   96	      self, single_action_spec: types.ActionSpec
   97	  ) -> types.Specs:
   98	    return dict(
   99	        logits=utils.get_logits_specs(single_action_spec),
  100	        y=types.ArraySpec((self._prediction_size,), jnp.float32),
  101	    )

  103	  def model_output_spec(
  104	      self, single_action_spec: types.ActionSpec
  105	  ) -> types.Specs:
  106	    return dict(
  107	        z=types.ArraySpec((self._prediction_size,), jnp.float32),
  108	        aux_pi=utils.get_logits_specs(single_action_spec),
  109	        q=types.ArraySpec((self._num_bins,), jnp.float32),
  110	    )
```
Comments:  
- `y` and `z` are treated as categorical logits later (via KL vs targets).  
- `q` is categorical value logits over `num_bins` bins.  

**3) The network factory and MLP torso + heads**

`disco_rl/disco_rl/networks/nets.py`:
```python
  32	def get_network(name: str, *args, **kwargs) -> types.PolicyNetwork:
  33	  """Constructs a network."""
  35	  def _get_net():
  36	    if name == 'mlp':
  37	      return MLP(*args, **kwargs)
  38	    else:
  39	      raise ValueError(f'Unknown network: {name}')

  47	  module_init_fn, one_step_fn = hk.without_apply_rng(
  48	      hk.transform_with_state(_agent_step)
  49	  )
  50	  _, unroll_fn = hk.without_apply_rng(hk.transform_with_state(_unroll))
```
  
```python
  59	class MLPHeadNet(hk.Module):
  75	    if model_out_spec:
  76	      self._model = action_models.get_action_model(
  77	          model_arch_name,
  78	          action_spec=action_spec,
  79	          out_spec=model_out_spec,
  80	          **model_kwargs,
  81	      )

  94	  def _head_pass(self, embedding: chex.Array) -> dict[str, chex.Array]:
  99	      output = hk.nets.MLP(
 100	          output_sizes=(np.prod(spec.shape),),
 101	          w_init=self._head_w_init,
 102	          name='torso_head',
 103	      )(embedding)

 115	  def __call__(...):
 118	    torso = self._embedding_pass(inputs)
 119	    out = self._head_pass(torso)
 120	    if self._model:
 121	      root = self._model.root_embedding(torso)
 122	      model_out = self._model.model_step(root)
 123	      out.update(model_out)
 124	    return out
```
  
  
```python
 152	  def _embedding_pass(...):
 156	    inputs = [hk.Flatten()(x) for x in jax.tree_util.tree_leaves(inputs)]
 157	    inputs = jnp.concatenate(inputs, axis=-1)
 158	    return hk.nets.MLP(self._dense, name='torso')(inputs)
```
Comments:  
- `_embedding_pass` is the observation-to-torso embedding mapping (flatten + concat + MLP).  
- `_head_pass` maps the torso embedding to the required outputs (`logits`, `y`).  
- If a `model_out_spec` exists, it adds the action-conditional model outputs (`z`, `aux_pi`, `q`) via `action_models`.  

**4) The action-conditional “model head”**

`disco_rl/disco_rl/networks/action_models.py`:
```python
  36	class LSTMModel:
  51	  def _model_transition_all_actions(self, embedding: hk.LSTMState) -> chex.Array:
  55	    num_actions = utils.get_num_actions_from_spec(self._action_spec)
  59	    one_hot_actions = jnp.eye(num_actions).astype(...)  # [A, A]
  62	    batched_one_hot_actions = jnp.tile(one_hot_actions, [batch_size, 1])  # [BA, A]
  66	    all_actions_embed = jax.tree.map(
  67	        lambda x: jnp.repeat(x, repeats=num_actions, axis=0), embedding
  68	    )  # [BA, *H]

  70	    lstm_output, _ = hk.LSTM(self._lstm_size, name='action_cond')(
  71	        batched_one_hot_actions, all_actions_embed
  72	    )
```

```python
  75	  def _model_head_pass(self, transition_output: chex.Array) -> dict[str, chex.Array]:
  84	    for key, pred_spec in self._out_spec.items():
  85	      pred = hk.nets.MLP(self._head_mlp_hiddens + (np.prod(pred_spec.shape),))(
  86	          transition_output
  87	      )
  88	      model_outputs[key] = pred.reshape(
  89	          (batch_size, num_actions, *pred_spec.shape)
  90	      )
```

```python
 100	  def root_embedding(self, state: chex.Array) -> hk.LSTMState:
 102	    flat_state = hk.Flatten()(state)
 103	    cell = hk.Linear(self._lstm_size)(flat_state)
 104	    return hk.LSTMState(hidden=jnp.tanh(cell), cell=cell)
```

Comments:  
- This is the code path that creates per-action outputs like `z[s,a]`, `aux_pi[s,a,*]`, `q[s,a,*]`.  
- It’s “LSTM-based” but not an across-time agent RNN;   
it’s a per-step action-conditional transition like in MuZero/Muesli.  

---

## Q2. How to train the Agent-network?

### Answer
Training happens in `Agent.learner_step(...)`:  
1) Collect a rollout (via repeated `actor_step`, often in a `jax.lax.scan` loop).  
2) Re-unroll the agent network on that rollout to get “current policy” outputs.  
3) Use the update rule’s meta-network (`unroll_meta_net`) to produce targets and auxiliary signals.  
4) Compute total loss (meta-target imitation losses + value/q loss).  
5) Compute gradients w.r.t. agent parameters with `jax.grad(...)`.  
6) Apply optax optimizer update (`Adam-like` rescaling + clipping + LR) to update agent params.  

### Source code (with line numbers) + comments

**Training step and parameter update**

`disco_rl/disco_rl/agent.py`:
```python
 248	  def learner_step(...):
 265	    eta_inputs = types.UpdateRuleInputs(
 270	        behaviour_agent_out=rollout.agent_outs,  # behavior policy outputs
 271	        agent_out=agent_out,  # current policy outputs (recomputed)
 272	        value_out=None,
 273	    )

 275	    # Apply the update network (meta-network) to produce targets/signals.
 276	    meta_out, new_meta_state = self.update_rule.unroll_meta_net(...)

 288	    # Differentiate the loss w.r.t. agent parameters.
 289	    dloss_dparams = jax.grad(self._loss, has_aux=True)
 290	    grads, (_, last_agent_net_state, logging_dict) = dloss_dparams(...)

 302	    updates, new_opt_state = self._optimizer.update(
 303	        grads, learner_state.opt_state, learner_state.params
 304	    )
 305	    new_params = optax.apply_updates(learner_state.params, updates)
```
Comments:  
- The rollout contains the behavior policy outputs (what the actor produced), but the learner re-unrolls the net to get consistent “current” outputs for training (`agent_out`).  
- Meta-network outputs (`meta_out`) act like “training targets” that define the agent loss.  
- The update uses optax (see also Q6).  

---

## Q3. How to generate predictions (π, y, z) from Agent-network? What are inputs/outputs? How to transform observations to inputs?

### Answer
**Inputs to the agent network**:  
- `observation` (batched, as returned by the environment).  
- `should_reset` mask (used by stateful nets; MLP ignores it).  
- parameters (+ optional Haiku state).  

**Outputs (“predictions”) from the agent network for DiscoRL**:  
- `π`: represented by `logits` (`π = softmax(logits)`).  
- `y`: a length-`Y` vector (used as categorical logits).  
- `z`: an action-conditional length-`Y` vector **for each action**;   
  `z_a` is selected for the taken action.  

**Observation → network input transform**:  
The default agent network is an MLP that:  
1) flattens each leaf of the observation pytree (e.g., board image),  
2) concatenates them along the feature axis,  
3) feeds the vector through an MLP torso.  

### Source code (with line numbers) + comments

**Agent forward pass + action sampling**

`disco_rl/disco_rl/agent.py`:
```python
 147	  def actor_step(...):
 155	    # Perform inference on the agent's network.
 156	    should_reset = timestep.step_type == dm_env.StepType.LAST
 157	    agent_outs, next_actor_state = self._network.one_step(
 158	        actor_params, actor_state, timestep.observation, should_reset
 159	    )
 160	    # Sample actions.
 161	    actions = distrax.Softmax(logits=agent_outs['logits']).sample(seed=rng)
```
Comments:  
- `agent_outs['logits']` are the policy logits (π after softmax).  
- `agent_outs` also contains `y`, `z`, `aux_pi`, `q` for DiscoRL (see Q1).  

**Observation flattening / embedding**

`disco_rl/disco_rl/networks/nets.py`:
```python
 152	  def _embedding_pass(...):
 156	    inputs = [hk.Flatten()(x) for x in jax.tree_util.tree_leaves(inputs)]
 157	    inputs = jnp.concatenate(inputs, axis=-1)
 158	    return hk.nets.MLP(self._dense, name='torso')(inputs)
```
Comments:
- This is the only observation preprocessing in the default agent net: flatten + concat.

**How `z_a` is obtained from per-action `z`**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 225	    # Parse the agent's output.
 229	    logits = agent_out['logits']
 230	    y = agent_out['y']
 231	    z = agent_out['z']
 232	    z_a = utils.batch_lookup(agent_out['z'], actions)
```
Comments:  
- `z` is stored as `[T, B, A, Y]` (time+batch+action+vector).  
- `batch_lookup` selects the slice for the executed action, producing `z_a` with shape `[T, B, Y]`.  

---

## Q4. How to calculate loss between Agent rollouts and targets from Meta-network?

### Answer
For DiscoRL, the agent loss is (per time-step):  
- `KL(π_hat || π)` where `π_hat` comes from meta-network and `π` is the agent’s `logits`.  
- `KL(y_hat || y)` where `y_hat` is meta target and `y` is agent prediction.  
- `KL(z_hat || z_a)` where `z_hat` is meta target and `z_a` is the agent’s action-selected `z`.  
- plus an auxiliary 1-step policy prediction loss: `KL(stop_grad(logits_{t+1}) || aux_pi_a)`.  

Additionally, there is a **Q/value loss** (`q_loss`) computed in `agent_loss_no_meta` (see Q26/Q36).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 234	    # Parse the meta-net's output.
 235	    pi_hat = meta_out['pi']
 236	    y_hat = meta_out['y']
 237	    z_hat = meta_out['z']
 238	    if not backprop:
 239	      pi_hat, y_hat, z_hat = jax.lax.stop_gradient((pi_hat, y_hat, z_hat))

 245	    pi_loss_per_step = rlax.categorical_kl_divergence(pi_hat, logits)
 246	    y_loss_per_step = rlax.categorical_kl_divergence(y_hat, y)
 247	    z_loss_per_step = rlax.categorical_kl_divergence(z_hat, z_a)

 249	    # Compute auxiliary 1-step policy prediction loss.
 250	    aux_pi = rollout.agent_out['aux_pi'][:-1]  # [T, B, A, A]
 251	    aux_pi_a = utils.batch_lookup(aux_pi, actions)  # [T, B, A]
 252	    aux_policy_target = rollout.agent_out['logits'][1:]  # [T, B, A]
 253	    aux_policy_loss_per_step = rlax.categorical_kl_divergence(
 254	        jax.lax.stop_gradient(aux_policy_target), aux_pi_a
 255	    )
```
Comments:  
- `categorical_kl_divergence(target_logits, pred_logits)` is used for all four losses;   
so `y`/`z` are treated as logits over a categorical distribution (same as the paper’s “vector predictions” trained by KL).  
- `backprop` controls whether gradients flow into meta targets (see Q37).  

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 296	  def agent_loss_no_meta(...):
 302	    """Value losses that do not interfere with meta-gradient."""
 306	    q_a = utils.batch_lookup(rollout.agent_out['q'], rollout.actions)[:-1]
 307	    value_loss_per_step = value_utils.value_loss_from_td(
 308	        value_net_out=q_a,
 309	        td=jax.lax.stop_gradient(td),
 310	        nonlinear_transform=True,
 311	        categorical_value=True,
 312	        max_abs_value=self._max_abs_value,
 313	    )
```
Comments:  
- This is where `q_loss`/value loss is computed, using TD targets from meta-side computations (`meta_out['q_td']`).  

---

## Q5. How to calculate gradients of Agent-network?

### Answer  
Gradients of the agent network are computed by differentiating the scalar training loss w.r.t. the agent parameters using `jax.grad(...)`.   
During multi-device training (`pmap`), gradients are averaged across devices using `jax.lax.pmean`.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/agent.py`:
```python
 288	    # Get gradient of the loss function using the latest rollout and parameters.
 289	    dloss_dparams = jax.grad(self._loss, has_aux=True)
 290	    grads, (_, last_agent_net_state, logging_dict) = dloss_dparams(
 291	        learner_state.params,
 292	        agent_net_state=agent_net_state,
 293	        meta_state=learner_state.meta_state,
 294	        rollout=rollout,
 295	        meta_out=meta_out,
 296	        is_meta_training=is_meta_training,
 297	    )
 298	    # Average gradients across the other learner devices involved in the `pmap`.
 299	    if self._batch_axis_name is not None:
 300	      grads = jax.lax.pmean(grads, axis_name=self._batch_axis_name)
```
Comments:  
- `_loss(...)` includes both the meta-target KL losses and the no-meta q/value loss (see `Agent._loss`, Q26).  

---

## Q6. How to update the parameters of Agent-network?

### Answer  
The agent uses an optax optimizer chain:  
1) `scale_by_adam_sg_denom()` (Adam-like rescaling, but with `stop_gradient` through the denominator),  
2) gradient clipping,  
3) multiply by `-learning_rate`,  
4) applies updates via `optax.apply_updates(params, updates)`.  

### Source code (with line numbers) + comments

**Optimizer definition**

`disco_rl/disco_rl/agent.py`:
```python
 101	    # Define the optimiser.
 102	    self._optimizer = optax.chain(
 103	        optimizers.scale_by_adam_sg_denom(),
 104	        optax.clip(max_delta=self.settings.max_abs_update),
 105	        optax.scale(-self.settings.learning_rate),
 106	    )
```

**Applying updates**

`disco_rl/disco_rl/agent.py`:
```python
 302	    updates, new_opt_state = self._optimizer.update(
 303	        grads, learner_state.opt_state, learner_state.params
 304	    )
 305	    # Update parameters.
 306	    new_params = optax.apply_updates(learner_state.params, updates)
```

**Adam-like rescaling that blocks meta-gradients through the variance term**

`disco_rl/disco_rl/optimizers.py`:
```python
  54	    updates = jax.tree.map(
  55	        lambda m, v: m / (jnp.sqrt(v) + eps),
  56	        mu_hat,
  57	        jax.lax.stop_gradient(nu_hat),  # NOTE: stop_gradient on nu_hat here
  58	    )
```
Comments:  
- This is important in meta-learning contexts: you typically don’t want the meta-gradient to “cheat” through optimizer statistics (unless you explicitly intend to).  

---

## Q7. What is the neural network architecture of the Meta-network? Where is it defined? How is it defined?

### Answer  
The DiscoRL “Meta-network” is an LSTM-based target generator implemented in Haiku:  

- It builds input features from rollout data (agent outputs, target agent outputs, rewards, terminals, advantage/TD features) via a configurable transform pipeline.  

- It uses:  
  1) a **per-trajectory LSTM** unrolled backwards (with reset masking) to produce a per-time hidden representation,  
  2) multiplicative conditioning on a **meta-RNN state** (“MetaLSTM”) that summarizes an agent’s learning progress across updates,  
  3) linear decoders to produce per-time targets `(pi_hat, y_hat, z_hat)`.  

The meta-network is defined in:  
- `disco_rl/disco_rl/networks/meta_nets.py` (the actual architecture),  
and is instantiated by:  
- `disco_rl/disco_rl/update_rules/disco.py` (wired into `DiscoUpdateRule` via `hk.transform_with_state`).  

### Source code (with line numbers) + comments

**Instantiation inside DiscoUpdateRule**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
  60	    # Meta-network.
  61	    def meta_net_fn(*args, **kwargs):
  62	      if net['name'] == 'lstm':
  63	        return meta_nets.LSTM(**net)(*args, **kwargs)
  67	    self._eta_init_fn, self._eta_apply = hk.transform_with_state(meta_net_fn)
```

**Meta-network forward pass (targets are produced here)**

`disco_rl/disco_rl/networks/meta_nets.py`:
```python
  45	class LSTM(MetaNet):
  78	  def __call__(self, inputs: types.UpdateRuleInputs, axis_name: str | None) -> types.UpdateRuleOuts:
  96	    # Construct inputs for the meta network.
 100	    x, policy_emb = _construct_input(...)

 109	    # Unroll the per-trajectory RNN core in reverse direction for bootstrapping.
 110	    per_trajectory_rnn_core = hk.ResetCore(hk.LSTM(self._hidden_size))
 112	    x, _ = hk.dynamic_unroll(..., reverse=True)

 119	    # Condition on (per-lifetime) meta RNN state via multiplicative interaction.
 120	    x = _multiplicative_interaction(
 121	        x=x,
 122	        y=self._meta_rnn_core.output(meta_rnn_state),
 123	        initializer=self._state_init,
 124	    )

 126	    meta_input_emb = hk.BatchApply(hk.Linear(1, w_init=self._output_init))(x)
 132	    y_hat = hk.BatchApply(hk.Linear(self._prediction_size, w_init=self._aux_init))(x)
 135	    z_hat = hk.BatchApply(hk.Linear(self._prediction_size, w_init=self._aux_init))(x)

 139	    # Compute the policy target (pi).
 140	    w = jnp.repeat(jnp.expand_dims(x, 2), num_actions, axis=2)
 141	    w = jnp.concatenate([w, policy_emb], axis=-1)
 142	    w = _conv1d_net(self._policy_target_channels)(w)
 143	    w = hk.BatchApply(hk.Linear(1, w_init=self._policy_target_init))(w)
 146	    pi_hat = jnp.squeeze(w, -1)  # [T, B, A]
```
Comments:  
- `reverse=True` in the per-trajectory RNN unroll is a design choice to enable bootstrapping-style conditioning from future parts of the trajectory.  
- `MetaLSTM` (next snippet) is the meta-RNN that summarizes learning dynamics.  

**Meta-RNN (MetaLSTM) update across the agent’s lifetime**

`disco_rl/disco_rl/networks/meta_nets.py`:
```python
 160	class MetaLSTM(hk.Module):
 179	  def unroll(...):
 192	    meta_inputs, _ = _construct_input(...)
 206	    x = jnp.concatenate(input_list, axis=-1)  # [T, B, ...]
 208	    x = _batch_mlp(self._embedding_size)(x)  # [T, B, E]
 210	    x_avg = x.mean(axis=(0, 1))  # [E]
 212	    if axis_name is not None:
 213	      x_avg = jax.lax.pmean(x_avg, axis_name=axis_name)
 216	    core = self._core_constructor()
 217	    _, new_state = core(x_avg, state)
 218	    return new_state
```
Comments:  
- The meta-RNN state update uses an **average pooled** embedding over time and batch to create a summary statistic for the whole rollout/update.  

---

## Q8. How to train a new Meta-network?

### Answer  
Meta-training (“train a new meta-network”) means   
learning the **update rule parameters** (`update_rule_params`, i.e., the meta-network weights)   
by optimizing an **outer/meta objective** through an inner learning process.  

In this repo’s Colab (`colabs/meta_train.ipynb`), the outer loop does roughly:  

1) For each agent in a population:  
   - run `num_inner_steps` inner updates of the agent using `Agent.learner_step(..., is_meta_training=True)`, so gradients can flow into `update_rule_params`,  
   - evaluate the adapted agent on validation data,  
   - compute a meta-loss on validation (policy gradient with advantage from a learned `ValueFunction` + regularizers),  
   - compute meta-gradient `d(meta_loss)/d(update_rule_params)` via `jax.grad`.  
  
2) Average meta-gradients across agents and apply a meta-optimizer step (optax Adam) to update `update_rule_params`.  

### Source code (with line numbers) + comments

**Inner update uses `is_meta_training=True`**

`disco_rl/colabs/meta_train.ipynb`:
```python
 308          def _inner_step(carry, inputs):
 313            # Update agent's parameters.
 314            new_learner_state, new_actor_state, metrics = agent.learner_step(
 319                update_rule_params=update_rule_params,
 320                is_meta_training=True,
```
Comments:
- This is what enables gradient flow into meta-network targets (see Q37).

**Outer loss + meta-gradient**

`disco_rl/colabs/meta_train.ipynb`:
```python
 338          def _outer_loss(
 348            # Perform N inner steps (i.e. agent's params' updates).
 351                jax.lax.scan(_inner_step, ..)
 ...
 374            # Calculate value_fn on the validation rollout.
 384            pg_loss_per_step = utils.differentiable_policy_gradient_loss(
 415            meta_loss = pg_loss_per_step.mean() + reg_loss
 ...
 438          # Calculate meta gradients.
 439          meta_grads, outputs = jax.grad(_outer_loss, has_aux=True)(..)
```

Comments:  
- The meta-loss here is a validation policy-gradient loss (plus regularizers), not the same as the inner Disco loss.  
- `jax.grad` differentiates *through* the inner updates (because inner used `is_meta_training=True`).  

**Meta-parameter update step**

`disco_rl/colabs/meta_train.ipynb`:
```python
 462   def meta_update(
 ...
 525     avg_meta_gradient = jax.tree.map(
 526         lambda x: x.mean(axis=0), utils.tree_stack(meta_grads)
 527     )

 528     meta_update, meta_opt_state = meta_opt.update(
 529         avg_meta_gradient, meta_opt_state
 530     )
 
 531     update_rule_params = optax.apply_updates(update_rule_params, meta_update)
```

---

## Q9. How to finetune a Meta-network? (Start from `disco_rl/colabs/meta_train.ipynb`)

### Answer  
Finetuning = meta-training starting from an existing checkpoint (e.g., Disco103 weights) instead of random initialization.  

In `meta_train.ipynb`, the finetuning switch is simply choosing the initial `update_rule_params`:  
- **train from scratch**: start with `random_update_rule_params`.    
- **finetune Disco103**: start with `disco_103_params` (downloaded earlier in the notebook).  

Then run the same meta-training loop (compute meta-gradients and apply meta-optimizer updates).  

### Source code (with line numbers) + comments

`disco_rl/colabs/meta_train.ipynb`:
```python
 572	 # Use random params for the update rule.
 573	 update_rule_params = random_update_rule_params  # can be `disco_103_params`
```
Comments:  
- To finetune, set `update_rule_params = disco_103_params` (keeping everything else the same).  

---

## Q10. How to generate targets (π_hat, y_hat, z_hat) from Meta-network? Inputs/outputs?

### Answer  
Targets `(π_hat, y_hat, z_hat)` are generated by:  
1) computing extra signals from value utilities (Retrace/V-trace, advantages, TDs, importance weights) using the agent’s outputs + a target (EMA) copy of agent params,  
2) packaging those signals into `rollout.extra_from_rule`,  
3) feeding the full `UpdateRuleInputs` to the meta-network (`meta_nets.LSTM`) to produce:  
   - `pi` → `π_hat` with shape `[T, B, A]`.  
   - `y` → `y_hat` with shape `[T, B, Y]`.  
   - `z` → `z_hat` with shape `[T, B, Y]`.  

Inputs the meta-network uses are configurable via `get_input_option()` (sources + transforms),   
but importantly it **does not directly consume raw observations** in the default configuration;   
it consumes rollouts’ agent outputs, rewards, terminals, and value-derived signals.    

### Source code (with line numbers) + comments

**Meta-network is called from DiscoUpdateRule.unroll_meta_net**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 132	    # Unroll the target policy.
 133	    target_out, _ = unroll_policy_fn(
 134	        meta_state['target_params'],  问: 这是agent params? meta-network params? agent-net的state?
 135	        state,
 136	        rollout.observations,
 137	        rollout.should_reset_mask_fwd,
 138	    )

 140	    # TD-value targets (Retrace over q-values here).
 141	    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(
 144	        q_net_out=rollout.agent_out['q'],
 145	        target_q_net_out=target_out['q'],
 ...
 161	    # Apply the meta-network.
 162	    rollout.extra_from_rule = dict(
 163	        v_scalar=value_outs.value,
 164	        adv=value_outs.adv,
 165	        normalized_adv=value_outs.normalized_adv,
 166	        q=value_outs.target_q_value,
 ...
 169	        target_out=target_out,
 170	    )
 171	    meta_out, new_rnn_state = self._eta_apply(
 172	        meta_params,
 173	        meta_state['rnn_state'],
 175	        rollout,
 176	        axis_name=axis_name,
 177	    )
 178	    chex.assert_rank(meta_out['pi'], 3)  # [T, B, A]
 179	    chex.assert_rank(meta_out['y'], 3)  # [T, B, Y]
 180	    chex.assert_rank(meta_out['z'], 3)  # [T, B, Y]
```
Comments:  
- `target_out` is produced by unrolling the agent net using `meta_state['target_params']` (an EMA “target network”).  
- `value_utils.get_value_outs` produces advantage/TD/etc signals and normalizes them via EMAs.  
- The meta-network (`_eta_apply`) consumes the rollout + `extra_from_rule`.  

**Meta-network’s outputs (pi_hat, y_hat, z_hat)**

`disco_rl/disco_rl/networks/meta_nets.py`:
```python
 131	    # Compute the y, z targets.
 132	    y_hat = hk.BatchApply(
 133	        hk.Linear(self._prediction_size, w_init=self._aux_init)
 134	    )(x)

 135	    z_hat = hk.BatchApply(
 136	        hk.Linear(self._prediction_size, w_init=self._aux_init)
 137	    )(x)
 ...
 146	    pi_hat = jnp.squeeze(w, -1)  # [T, B, A]

 149	    meta_out = dict(pi=pi_hat, y=y_hat, z=z_hat, meta_input_emb=meta_input_emb)
```

**What the meta-network uses as inputs is configured here**

`disco_rl/disco_rl/update_rules/disco.py` (excerpt from `get_input_option()`):
```python
 336	          types.TransformConfig(
                    source='agent_out/logits',
                    transforms=('drop_last', 'softmax', 'stop_grad', 'select_a'),
                  ),

 344	          types.TransformConfig(
                    source='rewards', 
                    transforms=('sign_log',)
                  ),

 345	          types.TransformConfig(
                    source='is_terminal',
                    transforms=('masks_to_discounts',),
                  ),
 
 353	          types.TransformConfig(
                    source='extra_from_rule/adv', 
                    transforms=('sign_log', 'stop_grad')
                  ),

 356	          types.TransformConfig(
                    source='extra_from_rule/normalized_adv', 
                    transforms=('stop_grad',)
                  ),
```
Comments:  
- The meta-net gets transformed versions of agent logits, rewards, terminal flags, advantages, etc.  
- Notice there is no `source='observations'` in the default option; i.e., meta-net is designed to be observation-agnostic in this harness.  

---

## Q11. How to calculate loss of meta-network?

### Answer  
In this repo’s meta-training setup, the meta-network (update rule parameters) is not trained by directly minimizing a supervised loss on its outputs.   
Instead, it is trained by minimizing an **outer loss** defined on the *post-update agent’s behavior* (validation rollout),   
and gradients flow into the meta-network parameters through the inner learning process.  

In `meta_train.ipynb`, the outer/meta loss is:  
- a policy gradient loss computed on validation rollouts using advantages from a learned `ValueFunction`,  
- plus several regularizers (entropy, target-KL, y/z entropy, etc.).  

### Source code (with line numbers) + comments

`disco_rl/colabs/meta_train.ipynb`:
```python
 374   # Calculate value_fn on the validation rollout.
 375   value_out, _, _, _ = agent.value_fn.get_value_outs(
 ...
 384   pg_loss_per_step = utils.differentiable_policy_gradient_loss(
 385       logits_on_valid, actions_on_valid, adv_t=adv_t, backprop=False
 386   )
 ...
 389   reg_loss = 0
 390   reg_loss += -1e-2 * distrax.Softmax(logits_on_valid).entropy().mean() # entr
 ...
 414   # Meta loss.
 415   meta_loss = pg_loss_per_step.mean() + reg_loss
```

Comments:  
- This is the objective whose gradient is taken w.r.t. `update_rule_params`.  
- The KL losses between meta targets and agent predictions (the Disco inner loss) are *inner-loop* losses used to update agent params, not the outer loss for meta params.  

---

## Q12. How to calculate gradient of meta-network?

### Answer  
Meta-gradients are computed as `jax.grad(outer_loss)(update_rule_params, ...)`, i.e. gradient of the outer objective w.r.t. the meta parameters (the meta-network weights).   
Because the inner loop uses `is_meta_training=True`, the computation graph includes the meta-network outputs/targets.  

### Source code (with line numbers) + comments

`disco_rl/colabs/meta_train.ipynb`:
```python
 438    # Calculate meta gradients.
 439    meta_grads, outputs = jax.grad(_outer_loss, has_aux=True)(
 440        update_rule_params, agent_state, train_rollouts, valid_rollout, rng
 441    )
```

---

## Q13. How to update the parameters of Meta-network?

### Answer
The meta-network parameters (`update_rule_params`) are updated using an optax optimizer (Adam in the notebook).   
The notebook averages meta-gradients across a population of agents, applies the optimizer update, then uses `optax.apply_updates`.  

### Source code (with line numbers) + comments

`disco_rl/colabs/meta_train.ipynb`:
```python
 528   meta_update, meta_opt_state = meta_opt.update(
 529       avg_meta_gradient, meta_opt_state
 530   )
 531   update_rule_params = optax.apply_updates(update_rule_params, meta_update)
```

---

## Q14. How to calculate “Advantage estimates”?

### Answer  
Advantages are computed inside `value_utils.get_value_outs(...)`:  
- If only **state values** are available: use **V-trace** via `rlax.vtrace_td_error_and_advantage`, and take `pg_advantage` as the advantage.  
- If **Q-values** are available (DiscoRL case): use **Retrace** via `rlax.general_off_policy_returns_from_q_and_v` to compute Q targets; then define advantage as `(q_target - V)` (in this implementation `adv = q_target - target_values[:-1]`).  

### Source code (with line numbers) + comments

**Where `get_value_outs` chooses V-trace vs Retrace**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 172	  # Get value targets and value outs
 173	  if q_values is not None:
 174	    # Estimate state-action values. (Retrace path)
 176	    value_outs = estimate_q_values(...)
 188	  else:
 189	    # Estimate state values. (V-trace path)
 191	    value_outs = estimate_values(...)
```

**V-trace advantage**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 282	  batch_vtrace_fn = jax.vmap(
 283	      functools.partial(rlax.vtrace_td_error_and_advantage, lambda_=lambda_),
 ...
 291	  vtrace_return = batch_vtrace_fn(...)
 ...
 306	  value_out = types.ValueOuts(
 307	      adv=vtrace_return.pg_advantage,
 ...
```

**Retrace advantage**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 365	  batch_retrace_fn = jax.vmap(
 367	          rlax.general_off_policy_returns_from_q_and_v,
 ...
 386	  q_target = jax.tree.map(
 387	      lambda q: batch_retrace_fn(q, target_values, r, d, c_t),
 388	      target_q_a,
 389	  )
 ...
 403	  adv = jax.tree_util.tree_leaves(q_target)[0] - target_values[:-1]
```
Comments:  
- In Retrace mode, the “advantage” is the Q-target for the taken action minus the baseline value.  

---

## Q15. What is the purpose of Advantage estimates? Where/when are they used? Inputs/Outputs?

### Answer  
Purpose: Advantage estimates provide a baseline-centered learning signal to update policies (reduce variance and incorporate bootstrapping / off-policy corrections).  

Where/when used in this repo:  
1) **Actor-Critic baseline update rule**: advantage is used directly in the policy gradient loss.  
2) **DiscoRL**: advantages are computed and then:  
   - fed as *inputs* to the meta-network (so it can generate better targets),  
   - optionally normalized via EMA and also fed to the meta-network,  
   - logged as part of `meta_out`.  
3) **Meta-training outer loss**: the notebook uses a separate `ValueFunction` to compute normalized advantages on validation data; those advantages are used in the outer policy gradient loss.  

Inputs/outputs:  
- Inputs: rewards, discounts/terminal flags, actions, and (current + behavior) policy logits to compute importance weights; plus either V(s) or Q(s,a) predictions (and target-network versions for bootstrapping).  
- Output: `ValueOuts.adv` (and `normalized_adv`) with shape `[T, B]` (or `[T, B, 1]` depending on config).  

### Source code (with line numbers) + comments

**Actor-Critic uses advantage in policy gradient**

`disco_rl/disco_rl/update_rules/actor_critic.py`:
```python
 192	    pg_advs = (
 193	        meta_out['normalized_advs'] if self._normalize_adv else meta_out['raw_advs']
 196	    )
 ...
 213	    pg_loss_per_step = utils.differentiable_policy_gradient_loss(
 214	        logits, actions, adv_t=pg_advs, backprop=False
 215	    )
```

**DiscoRL computes advantage and passes it to meta-net via `extra_from_rule`**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 140	    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(...)
 162	    rollout.extra_from_rule = dict(
 164	        adv=value_outs.adv,
 165	        normalized_adv=value_outs.normalized_adv,
 ...
```

**Outer/meta loss uses validation advantages from a learned ValueFunction**

`disco_rl/colabs/meta_train.ipynb`:
```python
 374        value_out, _, _, _ = agent.value_fn.get_value_outs(
 380        adv_t = jax.lax.stop_gradient(value_out.normalized_adv)
 384        pg_loss_per_step = utils.differentiable_policy_gradient_loss(
 385            logits_on_valid, actions_on_valid, adv_t=adv_t, backprop=False
```

---

## Q16. What is the function of the Value Function in `value_fn.py`? Why necessary? Why does it change? What is modified?

### Answer  
`ValueFunction` is a **separate learned baseline** used only during **meta-training** (outer loop). It approximates V(s) for the current policy distribution so that:  
- the outer objective can use a policy-gradient loss with reduced variance,  
- advantages can be normalized stably via running statistics (EMA).  

It changes continuously because it is trained online on rollouts from an evolving agent (policy changes during inner loop), so the value function must track the shifting return distribution.  

When it changes, the main thing being modified is:  
- the value network parameters (`value_state.params`) via gradient descent,  
- plus EMA statistics (`adv_ema_state`, `td_ema_state`) used for normalization.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_fn.py`:
```python
  31	class ValueFunction:
  34	  Used only in meta-training.
 ...
  56	    # Build value function network.
  57	    self._value_fn = nets.get_network(
  58	        config.net,
  59	        out_spec={'value': types.ArraySpec([1], jnp.float32)},
  60	        module_name='value_fn',
  61	        **config.net_args,
  62	    )
```

`disco_rl/disco_rl/value_fns/value_fn.py`:
```python
 141	  def update(...):
 158	    def value_loss_fn(...):
 165	      value_losses = value_utils.value_loss_from_td(
 166	          net_out[:-1], jax.lax.stop_gradient(value_outs.normalized_td)
 167	      )
 ...
 172	    (value_loss, ...), dv_dparams = jax.value_and_grad(value_loss_fn, has_aux=True)(...)
 ...
 183	    new_params = optax.apply_updates(value_state.params, update)
 185	    new_state = types.ValueState(
 186	        params=new_params,
 ...
 189	        adv_ema_state=adv_ema_state,
 190	        td_ema_state=td_ema_state,
 191	    )
```
Comments:  
- This is standard value-function learning: compute targets/TDs, compute value loss, take gradients, update params.  

---

## Q17. How to evaluate a meta-network? (Start from `disco_rl/colabs/eval.ipynb`)

### Answer  
Evaluation (“meta-evaluation”) in this repo means:  
- Fix `update_rule_params` to a discovered checkpoint (e.g., Disco103 weights),  
- Train a fresh agent on an environment using `Agent.learner_step(..., is_meta_training=False)` (so the meta targets are treated as constants; no meta-gradient),  
- Track returns / metrics.  

`eval.ipynb` implements a simple training loop with a FIFO replay buffer:  
1) unroll actors to collect rollouts and add to replay buffer,  
2) sample rollouts for learning and apply `learner_step_fn` (pmapped),  
3) compute and plot average returns.  

### Source code (with line numbers) + comments

**Set meta-params (Disco103)**

`disco_rl/colabs/eval.ipynb`:
```python
 292	        learner_state = agent.initial_learner_state(rng_key)
 293	        actor_state = agent.initial_actor_state(rng_key)
 294	        update_rule_params = disco_103_params
```

**Training loop uses `is_meta_training=False`**

`disco_rl/colabs/eval.ipynb`:
```python
 376          # Update agent's parameters on the samples from the buffer.
 380            learner_state, _, metrics = learner_step_fn(
 385                update_rule_params,
 386                False,  # is_meta_training
 387            )
```

Comments:  
- Here, Disco targets are used to train the agent, but gradients do not flow back into `update_rule_params`.  

---

## Q18. In DiscoRL, how are the concepts/components assembled and how does data flow?

### Answer  
At a high level (matching the harness):  

1) **Environment → Actor**  
   - The environment produces observation `o_t`.  
   - The actor runs the **agent network** to produce `logits_t` (policy), `y_t`, and per-action `z_t[a], aux_pi_t[a], q_t[a]`.  
   - The actor samples action `a_t ~ Softmax(logits_t)` and steps the env.  
   - It stores `(o_t, a_t, r_t, discount_t, logits_t, agent_outs_t, actor_state_t)` into an `ActorRollout`.  

2) **Rollout → Learner (inner-loop update)**  
   - Learner re-unrolls agent network on stored observations to compute “current-policy” outputs `agent_out`.  
   - Learner calls `DiscoUpdateRule.unroll_meta_net` which:  
     - builds a **target policy** via `target_params` (EMA of agent params),  
     - computes TD/advantages using **Retrace / V-trace** (`value_utils.get_value_outs`),  
     - feeds these as **meta-network inputs** to produce targets `(π_hat, y_hat, z_hat)`.  
   - Learner computes agent loss: KL divergences vs targets + auxiliary policy prediction + q/value loss.  
   - Learner computes gradients and updates agent parameters with optax.  

3) **Meta-training (outer-loop)**  
   - Inner-loop: the above agent update runs with `is_meta_training=True` so targets are differentiable.  
   - Outer-loop: evaluate adapted agent; compute outer policy-gradient loss using a separate `ValueFunction` to get advantages; take gradient w.r.t. update rule params and update meta-network weights.  

Where your listed terms fit:  
- **Agent**: `disco_rl/disco_rl/agent.py` (actor_step, learner_step).  
- **Meta-network**: `disco_rl/disco_rl/networks/meta_nets.py`, invoked by `DiscoUpdateRule.unroll_meta_net`.  
- **ValueFunction**: `disco_rl/disco_rl/value_fns/value_fn.py` (outer-loop baseline; meta-training only).  
- **Advantage, V-trace, Retrace, TD-error**: `disco_rl/disco_rl/value_fns/value_utils.py`.  
- **Agent-LSTM**: the default agent net here is an MLP (stateless), but the API supports stateful nets; the LSTM used by the provided Disco agent is the *action-conditional* model head (see next bullet).  
- **action-model**: `disco_rl/disco_rl/networks/action_models.py` (action-conditional LSTM producing per-action outputs z/aux_pi/q).  
- **env-model**: not explicitly modeled in this minimal harness (no learned observation/reward transition model); the “model-like” component present is the action-conditional latent transition used to produce z/aux_pi/q.  
- **Meta-LSTM / Meta-RNN**: `MetaLSTM` inside `meta_nets.py` (persistent state over learning updates).  
- **inner-loop / outer-loop**: implemented in `colabs/meta_train.ipynb` (inner = learner_step; outer = meta_update).  
- **Actor-Critic**: provided as a baseline update rule (`update_rules/actor_critic.py`); also the meta-training outer loss uses policy-gradient style.  
- **Q_value**: `q` is the agent’s categorical value prediction per action; targets via Retrace.
- **aux_pi**: agent’s per-action 1-step policy prediction.  
- **(π, y, z)**: `logits` (π), `y` (obs-conditional), `z` (action-conditional) predictions.  

### Source code (with line numbers) + comments

For the “assembly”, these are the key glue points:  

`disco_rl/disco_rl/agent.py` (actor → rollout; learner uses update rule):
```python
 147	  def actor_step(...):
 157	    agent_outs, next_actor_state = self._network.one_step(...)
 161	    actions = distrax.Softmax(logits=agent_outs['logits']).sample(seed=rng)
```
```python
 275	    # Apply the update network.
 276	    meta_out, new_meta_state = self.update_rule.unroll_meta_net(...)
 289	    grads = jax.grad(self._loss, has_aux=True)(...)
 306	    new_params = optax.apply_updates(...)
```

`disco_rl/disco_rl/update_rules/disco.py` (value utils + meta targets + target params EMA):
```python
 132	    target_out, _ = unroll_policy_fn(meta_state['target_params'], ...)
 141	    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(...)
 171	    meta_out, new_rnn_state = self._eta_apply(..., rollout, ...)
 200	    coeff = hyper_params['target_params_coeff']
 202	    new_meta_state['target_params'] = jax.tree.map(lambda old, new: old * coeff + (1.0 - coeff) * new, ...)
```

---

## Q19. The three neural networks (agent, meta-network, value function): how do they interact and how does data flow?

### Answer
1) **Agent network** produces:  
   - behavior policy outputs during acting (stored in rollout),  
   - “current” outputs during learning (recomputed by unrolling).  

2) **Meta-network** consumes:  
   - agent outputs (policy logits, y, z, etc.),  
   - behavior outputs (for off-policy info),  
   - reward/terminal signals,  
   - value-derived signals (adv, TD, target policy outputs),  
   and produces **targets** `(π_hat, y_hat, z_hat)` that define the agent’s training loss.  

3) **Value function** (meta-training only) consumes:  
   - observations + target logits (policy) for a rollout,  
   and produces advantages/TDs for the *outer* meta-loss. It is trained online to track returns for the changing policy distribution.  

So data flows as:  
`env → agent_net → rollout → value_utils (+ target policy) → meta_net → targets → agent_loss → agent_param_update`.    
and during meta-training:  
`(inner updates) → validation rollout → value_fn advantages → outer loss → meta_param_update`.  

### Source code (with line numbers) + comments

- Agent network ↔ meta-network wiring: `Agent.learner_step` → `DiscoUpdateRule.unroll_meta_net` → meta targets (see `agent.py:275-286` and `disco.py:132-180`).  
- Meta-network’s outputs drive agent loss: `disco.py:234-257` (KL losses).  
- Value function is used in meta-training outer loss: `meta_train.ipynb:374-386`.  

---

## Q20. What is the relationship between EMA and Normalize? Are they equivalent? When should EMA vs normalize be used?

### Answer
They are not equivalent:  
- **EMA (exponential moving average)** is a *stateful estimator* that tracks running statistics (mean and variance / second moment) over time.  
- **Normalization** uses those statistics to transform a signal (e.g., advantage) into a standardized scale (e.g., roughly zero-mean, unit-variance).  

In this repo there are *two* related implementations:  
1) `utils.MovingAverage` (used for advantage/TD normalization in `value_utils.get_value_outs` and in `DiscoUpdateRule` meta_state).  
2) `input_transforms.EmaNorm` (a Haiku module used as an input transform option inside the meta-network input pipeline).  

When to use:  
- Use EMA-based normalization when your signal distribution is non-stationary (policy/value targets change over training) and you want stable scaling without computing full-dataset statistics.  
- Use plain normalization (mean/std of current batch) if you want fast, stateless normalization and can tolerate batch-to-batch noise (this repo mostly uses EMA to reduce noise and stabilize meta-gradients).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/utils.py` (EMA state + normalization):
```python
 173	class MovingAverage:
 203	  def update_state(...):
 217	      mean = jnp.mean(value)
 221	      return self._decay * moment + (1.0 - self._decay) * mean
 ...
 252	  def normalize(...):
 264	        return (val - mean) / (jnp.sqrt(var + root_eps) + self._eps)
```

`disco_rl/disco_rl/value_fns/value_utils.py` (advantage normalization driven by EMA):
```python
 208	  if adv_ema_state is not None and adv_ema_fn is not None:
 209	    new_adv_ema_state = adv_ema_fn.update_state(advantages, adv_ema_state, axis_name)
 212	    normalized_adv = adv_ema_fn.normalize(advantages, new_adv_ema_state)
```

`disco_rl/disco_rl/update_rules/input_transforms.py` (EMA normalization as a meta-network input transform):
```python
  58	class Normalize(InputTransform, hk.Module):
  63	    return EmaNorm(decay_rate=0.99, eps=1e-6, axis=(0, 1), cross_replica_axis=axis)(x)
```

---

## Q21. How are Advantage estimates calculated? Where are they calculated? Inputs/Outputs?

### Answer
Same as Q14/Q15, but explicitly:  
- Calculated in `disco_rl/disco_rl/value_fns/value_utils.py`, inside `get_value_outs(...)` via either `estimate_values` (V-trace) or `estimate_q_values` (Retrace).  
- Inputs include rollout rewards/discounts/actions, online policy logits `pi_logits`, behavior policy logits `mu_logits`, and either V(s) or Q(s,a) predictions (and optional target-network versions).  
- Output is in `types.ValueOuts.adv` (and `normalized_adv`), typically shaped `[T, B]` (or `[T, B, 1]` depending on whether you keep a singleton dim).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
  35	def get_value_outs(...):
 ...
 166	  rho = importance_weight(..., actions)
 ...
 173	  if q_values is not None:
 176	    value_outs = estimate_q_values(...)
 188	  else:
 191	    value_outs = estimate_values(...)
```

---

## Q22. What do Current, Behavior, and Target policies refer to respectively?

### Answer
In this harness:  
- **Behavior policy (μ)**: the policy that generated the rollout data (the actor’s logits stored in the rollout). Used to compute importance weights for off-policy corrections.  
- **Current policy (π)**: the online policy computed from the learner’s current parameters when learning (the re-unrolled `rollout.agent_out['logits']` passed into value utils as `pi_logits`).  
- **Target policy**: an EMA (“target network”) copy of agent parameters stored in `meta_state['target_params']`, used for bootstrapping targets (e.g., `target_out['q']`).  

### Source code (with line numbers) + comments

**Behavior logits come from actor rollouts (ActorTimestep stores logits)**

`disco_rl/disco_rl/types.py`:
```python
 143	@chex.dataclass
 144	class ActorTimestep:
 ...
 151	  agent_outs: AgentOuts
 152	  states: HaikuState
 153	  logits: Any
```

**Value utils picks μ differently depending on whether it gets ActorRollout or UpdateRuleInputs**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 119	  rewards, actions = rollout.rewards, rollout.actions
 120	  if isinstance(rollout, types.ActorRollout):
 121	    env_discounts, mu_logits = rollout.discounts, rollout.logits
 122	  else:
 125	    mu_logits = rollout.behaviour_agent_out['logits']
```

**Target policy from EMA parameters**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 132	    # Unroll the target policy.
 133	    target_out, _ = unroll_policy_fn(
 134	        meta_state['target_params'],
 ...
 200	    # Update target params (EMA).
 201	    coeff = hyper_params['target_params_coeff']
 202	    new_meta_state['target_params'] = jax.tree.map(
 203	        lambda old, new: old * coeff + (1.0 - coeff) * new,
```

---

## Q23. Should the shape of z in the agent's rollouts be [T, B, 600] or [T, B, 600, A]?

### Answer
In this implementation, `z` is **action-conditional**, so it is stored with an action dimension:  
- `z` in agent outputs has shape `[T+1, B, A, 600]` (time-major rollout, including bootstrap step).  
- The action-selected `z_a` used in the loss has shape `[T, B, 600]`.  

So it’s neither `[T, B, 600]` (that’s `z_a`), nor `[T, B, 600, A]` (wrong dimension order).   
It is `[T, B, A, 600]` (or `[T+1, B, A, 600]` for bootstrapped rollouts).    

### Source code (with line numbers) + comments

**Why action-conditional outputs get an extra action dimension**

`disco_rl/disco_rl/update_rules/base.py`:
```python
  66	  num_actions = utils.get_num_actions_from_spec(action_spec)
  68	  for key, val in model_out_spec.items():
  69	    agent_out_spec[key] = ArraySpec((num_actions, *val.shape), val.dtype)
```

**DiscoRL defines `z` as a model output spec (action-conditional)**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 103	  def model_output_spec(...):
 106	    return dict(
 107	        z=types.ArraySpec((self._prediction_size,), jnp.float32),
```

**Action model reshapes to `[B, A, ...]`**

`disco_rl/disco_rl/networks/action_models.py`:
```python
  88	      model_outputs[key] = pred.reshape(
  89	          (batch_size, num_actions, *pred_spec.shape)
  90	      )
```

**Loss uses `z_a` after selecting the taken action**

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 232	    z_a = utils.batch_lookup(agent_out['z'], actions)
```

---

## Q24. Why do many calculations require learner_state, actor_state, and meta_state as input? Is it because LSTM needs state?

### Answer
Partly yes, but it’s broader than “because LSTM”:  
- `learner_state` holds **agent params** and **optimizer state** (needed to apply updates).  
- `actor_state` holds **agent network recurrent/stateful state** (Haiku state).   
For the default MLP it may be empty, but the API supports stateful nets;   
and rollouts store `states` so you can unroll training consistently.  
- `meta_state` is the update rule’s state, and in DiscoRL it includes:  
  - meta-network RNN state (`rnn_state`) (this *is* an LSTM state),  
  - EMA states for advantage/TD normalization,  
  - target params (EMA copy of agent params) for bootstrapping.  

So, “states everywhere” is necessary to   
(1) correctly handle stateful models, and   
(2) maintain the meta-learning machinery (target network + running stats).    

### Source code (with line numbers) + comments

`disco_rl/disco_rl/agent.py`:
```python
  38	@chex.dataclass(frozen=True)
  39	class LearnerState:
  42	  params: hk.Params
  43	  opt_state: optax.OptState
  44	  meta_state: types.MetaState
```

`disco_rl/disco_rl/update_rules/disco.py` (meta_state contents):
```python
  87	    meta_state = dict(
  88	        rnn_state=meta_rnn_state,
  89	        adv_ema_state=self._adv_ema.init_state(),
  90	        td_ema_state=self._td_ema.init_state(),
  91	        target_params=params,
  92	    )
```

`disco_rl/disco_rl/types.py` (actor timestep includes `states`):
```python
 144	class ActorTimestep:
 152	  states: HaikuState
```

---

## Q25. Why do return values for agent predictions and meta-targets still contain various states?

### Answer
Because the next computation step often needs them:  
- Actor loop needs `actor_state` to continue stepping the policy (for stateful nets).  
- Learner needs an initial `agent_net_state` to unroll the agent network consistently over a rollout.  
- Meta-network needs and updates `meta_state['rnn_state']` (MetaLSTM) and EMA normalization states.  

Even when states are empty (stateless MLP), the API stays uniform.

### Source code (with line numbers) + comments

`disco_rl/disco_rl/agent.py` (actor_step returns next state; learner_step returns updated meta_state too):
```python
 157	    agent_outs, next_actor_state = self._network.one_step(...)
 ...
 173	    return actor_timestep, next_actor_state
```
```python
 276	    meta_out, new_meta_state = self.update_rule.unroll_meta_net(...)
 ...
 313	    learner_state = LearnerState(
 314	        params=new_params, opt_state=new_opt_state, meta_state=new_meta_state
 315	    )
 316	    return learner_state, last_agent_net_state, logging_dict
```

---

## Q26. Why is q-loss missing from agent loss? Agent loss includes π-loss, y-loss, z-loss, aux_pi-loss, but why is q-loss absent?

### Answer
`q_loss` is not missing; it is computed in a separate function `agent_loss_no_meta(...)` and then added to the total loss in `Agent._loss(...)`.  

The separation is deliberate:  
- `agent_loss` is the part that depends on meta targets; during meta-training it may allow gradients to flow into meta params.  
- `agent_loss_no_meta` is intended to **not interfere with meta-gradients** (it uses `stop_gradient` on TD targets), hence the name (see Q36).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/agent.py`:
```python
 234	    loss_per_step, log = self.update_rule.agent_loss(..., backprop=is_meta_training)
 237	    loss_per_step_no_meta, log_no_meta = self.update_rule.agent_loss_no_meta(...)
 241	    total_loss_per_step = loss_per_step + loss_per_step_no_meta
```

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 296	  def agent_loss_no_meta(...):
 302	    """Value losses that do not interfere with meta-gradient."""
```

---

## Q27. Difference between aux_pi_a and aux_pi? Difference between q_a and q? What does batch_lookup() do?

### Answer
- `aux_pi` is the **action-conditional** next-policy prediction logits for *all actions*: shape `[T, B, A, A]`.  
  -- First `A`: “conditioning action” dimension (which action you imagine taking now).  
  -- Second `A`: predicted distribution over next action.    

- `aux_pi_a` selects the conditioning action to be the actually taken action `a_t`, giving shape `[T, B, A]` (a single predicted next policy distribution per timestep).  

- `q` is the **action-conditional** categorical value logits for all actions: shape `[T, B, A, num_bins]`.  
- `q_a` selects the value logits for the taken action: shape `[T, B, num_bins]`.  

`batch_lookup(table, index)` performs a batched gather along the action dimension,   
i.e., “select `table[..., a_t, ...]` for each batch element/time”.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 250	    aux_pi = rollout.agent_out['aux_pi'][:-1]  # [T, B, A, A]
 251	    aux_pi_a = utils.batch_lookup(aux_pi, actions)  # [T, B, A]
```

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 306	    q_a = utils.batch_lookup(rollout.agent_out['q'], rollout.actions)[:-1]
```

`disco_rl/disco_rl/utils.py`:
```python
  50	def batch_lookup(table: chex.Array, index: chex.Array, num_dims: int = 2) -> chex.Array:
  55	    return jax.vmap(lambda x, i: x[i])(table, index)
  59	  return hk.BatchApply(_lookup, num_dims=num_dims)(table, index)
```
Comments:  
- `hk.BatchApply(..., num_dims=2)` means it batch-applies over the leading dims (typically `[T, B]`) and does the `vmap` gather per element.  

---

## Q28. Roles of Q-values and State-values? What is Value-net / Q-net? How defined? How do V-trace and Retrace work? Why are Q and V interchangeable here? Differences between value types?

### Answer
Roles:  
- **State value V(s)** estimates expected return from a state (baseline; used for advantage and bootstrapping).  
- **Action value Q(s,a)** estimates expected return if you take action `a` in `s` (used for control/value-based learning; can produce V via expectation under a policy).  

In this codebase:  
- “Value-net” = any network outputting `value` (or `v`) per state. Example: `ValueFunction` builds a net with `out_spec={'value': [1]}` (`value_fn.py`).  
- “Q-net” = a network outputting `q` per action. In DiscoRL, `q` is produced by the agent network’s action-conditional model head (`DiscoUpdateRule.model_output_spec` + `action_models.LSTMModel`).  

V-trace:  
- Used when you have V(s) predictions.  
- Computes corrected returns and a policy-gradient advantage under off-policy data using importance weights `rho`.  
- Inputs: target V(s_t), target V(s_{t+1}), rewards, discounts, rho, lambda.  
- Outputs (in this code): `value_target`, `td`, and `adv = pg_advantage`.  

Retrace:  
- Used when you have Q(s,a) (and optionally V(s)).  
- Computes off-policy corrected Q targets using clipped importance weights and trace coefficients.  
- Inputs: Q(s_t,a_t) (chosen action), V(s_t), rewards, discounts, `c_t = lambda * min(rho, 1)`.  
- Outputs: `q_target`, `q_td`, and `adv = q_target - V`.  

Why “interchangeable” in this implementation:  
- V(s) can be derived from Q(s,a) and π(a|s) by expectation: `V(s) = Σ_a π(a|s) Q(s,a)`.  
- When `value_net_out` is missing, `value_utils.extract_scalar_values_from_net_out` computes V from Q (policy-weighted sum).  
- The opposite direction (deriving Q purely from V) is not done; instead, if Q is missing, this code returns dummy Q fields.  

The four value variants in the question map roughly to:  
- `value_net_out` → state value logits/scalar from the online net.  
- `q_net_out` → action value logits/scalar from the online net.  
- `target_value_net_out` / `target_q_net_out` → bootstrapping values from a target/EMA net (or provided target outputs).  

### Source code (with line numbers) + comments

**Where V is derived from Q if needed**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 485	  if values is None:
 486	    # Get state values from Q-values if values are not explicitly given
 487	    pi_tree = jax.tree.map(jax.nn.softmax, pi_logits)
 488	    values = jax.tree.map(
 489	        lambda p, q: jnp.sum(p * q, axis=2), pi_tree, q_values
 490	    )
```

**V-trace implementation path**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 252	def estimate_values(...):
 282	  batch_vtrace_fn = jax.vmap(
 283	      functools.partial(rlax.vtrace_td_error_and_advantage, lambda_=lambda_),
 ...
 291	  vtrace_return = batch_vtrace_fn(...)
 298	  value_target = vtrace_return.errors + target_values[:-1]
 307	      adv=vtrace_return.pg_advantage,
```

**Retrace implementation path**

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 323	def estimate_q_values(...):
 365	  batch_retrace_fn = jax.vmap(
 366	      functools.partial(
 367	          rlax.general_off_policy_returns_from_q_and_v,
 368	          stop_target_gradients=True,
 369	      ),
 ...
 383	  clipped_rho = jnp.minimum(rho, 1.0)
 384	  lambda_rho = lambda_ * clipped_rho
 385	  c_t = lambda_rho
 386	  q_target = jax.tree.map(
 387	      lambda q: batch_retrace_fn(q, target_values, r, d, c_t),
 388	      target_q_a,
 389	  )
 ...
 403	  adv = jax.tree_util.tree_leaves(q_target)[0] - target_values[:-1]
 405	  q_td = jax.tree.map(lambda target, q: target - q, q_target, q_a)
```

---

## Q29. Is this understanding correct? Value target = V(s) + TD_error; TD error = target - prediction; Advantage = V-trace_return - V(s)

### Answer
Mostly consistent with how this code constructs targets:  
- In `estimate_values` (V-trace path), the code sets:  
  -- `value_target = vtrace_return.errors + target_values[:-1]`.  
  -- `td = value_target - values[:-1]`.  
  So “TD error = target - prediction” matches `td = value_target - values`.  

However, the “advantage” used for policy gradients is `vtrace_return.pg_advantage` (RLax’s V-trace PG advantage),     
which is related to (v-trace return − baseline) but is not literally named “V-trace_return” in this code.     
So your formula is directionally right, but the exact object used is RLax’s `pg_advantage`.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 298	  value_target = vtrace_return.errors + target_values[:-1]
 ...
 312	      td=value_target - values[:-1],
 307	      adv=vtrace_return.pg_advantage,
```

---

## Q30. Can we interpret state-value as predicted game score, Q-value as sum of predicted future rewards, and target_* as meta-network predicted scores?

### Answer
The *spirit* of the first part is correct, with a clarification:  
- V(s) and Q(s,a) estimate **expected discounted return**, which can correlate with “score”, but is formally the discounted sum of future rewards under a policy.  
- Q-values are not necessarily “next 10 or 20 moves”; the horizon is effectively the episode (discounted), unless the environment/algorithm imposes a finite horizon.  

The last part is **not correct in this repo**:  
- `target_value_net_out` / `target_q_net_out` are *not* meta-network predictions.  
- They are predictions from a **target/EMA copy of the agent network** (a target policy/value head), used for bootstrapping. In DiscoRL, the target outputs are computed by unrolling the agent net with `meta_state['target_params']`.  
- The **meta-network outputs are** `(pi_hat, y_hat, z_hat)` (targets for the agent’s policy/y/z predictions).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/disco.py` (target outputs are from target agent params, not meta-net):
```python
 132	    # Unroll the target policy.
 133	    target_out, _ = unroll_policy_fn(
 134	        meta_state['target_params'],
```

`disco_rl/disco_rl/networks/meta_nets.py` (meta-net outputs are pi_hat/y_hat/z_hat):
```python
 149	    meta_out = dict(pi=pi_hat, y=y_hat, z=z_hat, meta_input_emb=meta_input_emb)
```

---

## Q31. What does "importance weights" mean? What does rho mean? What is mu? (See value_utils.py:495-509)

### Answer
“Importance weights” correct for the mismatch between:  
- **behavior policy** μ(a|s): the policy that generated the data.  
- **target/current policy** π(a|s): the policy you want to evaluate/update.  

In off-policy learning, the basic importance sampling ratio is:
`rho_t = π(a_t|s_t) / μ(a_t|s_t) = exp(log_pi - log_mu)`.  

In this implementation:  
- `pi_logits` represent π.  
- `mu_logits` represent μ.  
- `rho` is computed from their log-probabilities of the taken actions, and `stop_gradient` is applied to rho (treating it as a constant for backprop stability).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 495	def importance_weight(
 500	  """Calculate importance weights from logits."""
 501	  log_prob_fn = lambda t, a: distrax.Softmax(t).log_prob(a)
 502	  log_pi_a_tree = jax.tree.map(log_prob_fn, pi_logits, actions)
 503	  log_mu_a_tree = jax.tree.map(log_prob_fn, mu_logits, actions)
 505	  # Joint probs.
 506	  log_pi_a = sum(jax.tree_util.tree_leaves(log_pi_a_tree))
 507	  log_mu_a = sum(jax.tree_util.tree_leaves(log_mu_a_tree))
 508	  rho = jax.lax.stop_gradient(jnp.exp(log_pi_a - log_mu_a))
 509	  return rho
```

---

## Q32. What is the purpose of normalized_adv? (Advantage normalization)

### Answer
`normalized_adv` is advantage scaled/centered using running (EMA) estimates of mean and variance. Purpose:
- stabilize learning by keeping the advantage scale consistent,  
- reduce sensitivity to reward scale differences across environments/agents,  
- improve meta-gradient stability (especially important in meta-learning).  

In this repo, `normalized_adv` is used:  
- as an optional signal for actor-critic/policy-gradient updates,  
- as an explicit **input feature** to the DiscoRL meta-network (`extra_from_rule/normalized_adv`).  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_utils.py`:
```python
 208	  if adv_ema_state is not None and adv_ema_fn is not None:
 209	    new_adv_ema_state = adv_ema_fn.update_state(
 210	        advantages, adv_ema_state, axis_name
 211	    )
 212	    normalized_adv = adv_ema_fn.normalize(advantages, new_adv_ema_state)
```

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 164	        adv=value_outs.adv,
 165	        normalized_adv=value_outs.normalized_adv,
```

`disco_rl/disco_rl/update_rules/disco.py` (meta-net input option includes normalized_adv):
```python
 356	          types.TransformConfig(
 357	              source='extra_from_rule/normalized_adv', transforms=('stop_grad',)
 358	          ),
```

---

## Q33. How are Advantage estimates calculated? Where? Inputs/Outputs? How are average action results or game score calculated?

### Answer
Advantages: see Q14/Q21 (computed in `value_utils.get_value_outs`).  

“Average action results / game score” in the provided evaluation notebook is computed from episodic returns:  
- The notebook accumulates rewards until episode termination (`discount == 0`),  
- then computes `total_returns` and divides by number of episodes to get `avg_returns`.  

### Source code (with line numbers) + comments

`disco_rl/colabs/eval.ipynb`:
```python
 210      def accumulate_rewards(acc_rewards, x):
 ...
 213        def _step_fn(acc_rewards, x):
 215          acc_rewards += rewards
 216          return acc_rewards * discounts, acc_rewards
```

`disco_rl/colabs/eval.ipynb`:
```python
 411      total_returns = (all_returns * (1 - all_discounts)).sum(axis=(1, 2))
 412      total_episodes = (1 - all_discounts).sum(axis=(1, 2))
 413      avg_returns = total_returns / total_episodes
```

---

## Q34. How is the Value Function modified? Why is it continuously modified?

### Answer  
It’s modified by gradient descent on a value loss computed from TD targets (V-trace style), and its EMA normalization statistics are updated each step. It is continuously modified because:  
- the policy/rollout distribution changes as the agent learns (especially across inner-loop updates),  
- so the value baseline must adapt online to remain accurate and useful for advantage estimation.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/value_fns/value_fn.py`:
```python
 172	    (value_loss, ...), dv_dparams = jax.value_and_grad(...)(...)
 179	    update, new_opt_state = self._value_opt.update(...)
 183	    new_params = optax.apply_updates(value_state.params, update)
```

---

## Q35. What is the role of the actor-critic algorithm in DiscoRL? In which stage does it operate? What data does it process?

### Answer
In this repository, “Actor-Critic” appears in two ways:  

1) As a **baseline update rule** (`ActorCritic` in `update_rules/actor_critic.py`) that you can choose instead of Disco.   
In that mode, actor-critic is the *inner-loop* agent learning rule:   
it processes rollouts, computes advantages/TDs (via `value_utils.get_value_outs`),   
and updates the agent policy/value heads via policy gradient + value loss + entropy regularization.  

2) In the **meta-training outer loss** (in `meta_train.ipynb`),   
the outer objective is also policy-gradient style (actor-critic-like),   
using a learned value baseline to compute advantages on validation data.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/actor_critic.py`:
```python
 146	    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(...)
 ...
 213	    pg_loss_per_step = utils.differentiable_policy_gradient_loss(...)
```

`disco_rl/colabs/meta_train.ipynb`:
```python
 384	    pg_loss_per_step = utils.differentiable_policy_gradient_loss(...)
```

---

## Q36. What does no_meta mean in agent_loss_no_meta()?

### Answer
`no_meta` means: “this part of the loss is designed not to interfere with the meta-gradient”.  

Concretely:  
- `agent_loss_no_meta` computes a value/q loss using TD targets,   
but uses `stop_gradient` on the TD signal,   
and the DiscoUpdateRule docstring explicitly calls it “value losses that do not interfere with meta-gradient”.  
- This separation helps ensure meta-optimization focuses on the meta-network’s target generation rather than exploiting value-loss gradients through targets/normalization.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 296	  def agent_loss_no_meta(...):
 302	    """Value losses that do not interfere with meta-gradient."""
 ...
 309	        td=jax.lax.stop_gradient(td),
```

`disco_rl/disco_rl/agent.py`:
```python
 234	    loss_per_step, log = self.update_rule.agent_loss(..., backprop=is_meta_training)
 237	    loss_per_step_no_meta, log_no_meta = self.update_rule.agent_loss_no_meta(...)
 241	    total_loss_per_step = loss_per_step + loss_per_step_no_meta
```

---

## Q37. Meta-network targets: stopped or not stopped? What does it mean when backprop=True and gradients flow through targets?

### Answer  
“Stopping targets” means applying `jax.lax.stop_gradient` to `(pi_hat, y_hat, z_hat)` before computing the agent loss. In that case:  
- agent parameters still get gradients (because the loss depends on agent outputs),  
- but meta-network parameters **do not** get gradients through those targets (targets treated as constants).  

“Not stopping targets” (i.e., `backprop=True`) means:  
- the agent loss is differentiable w.r.t. the meta targets,  
- therefore meta-network parameters can receive gradients through how they generated `(pi_hat, y_hat, z_hat)`.  
This is required for meta-training (outer-loop) where you want to learn the update rule itself.  

In this code:  
- evaluation / standard training sets `is_meta_training=False` → targets are stop-grad’ed.  
- meta-training sets `is_meta_training=True` → targets are not stop-grad’ed.  

### Source code (with line numbers) + comments

`disco_rl/disco_rl/update_rules/disco.py`:
```python
 234	    # Parse the meta-net's output.
 235	    pi_hat = meta_out['pi']
 236	    y_hat = meta_out['y']
 237	    z_hat = meta_out['z']
 238	    if not backprop:
 239	      pi_hat, y_hat, z_hat = jax.lax.stop_gradient((pi_hat, y_hat, z_hat))
```

`disco_rl/disco_rl/agent.py`:
```python
 234	    loss_per_step, log = self.update_rule.agent_loss(
 235	        eta_inputs, meta_out, hyper_params, backprop=is_meta_training
 236	    )
```

`disco_rl/colabs/meta_train.ipynb` (inner loop sets `is_meta_training=True`):
```python
 314        new_learner_state, new_actor_state, metrics = agent.learner_step(
 320            is_meta_training=True,
```
