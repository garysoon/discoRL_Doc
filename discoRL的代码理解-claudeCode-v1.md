# DiscoRL代码理解：Agent和Meta-Network的训练机制

## 1. 整体架构概述

DiscoRL（Discovered Reinforcement Learning）采用**元学习（Meta-Learning）**的方式来自动发现强化学习算法。其核心包含三个主要组件：

1. **Agent（智能体）**：执行环境交互的策略网络
2. **Meta-Network（元网络）**：基于LSTM的网络，用于生成训练目标
3. **Meta-Value Function（元价值函数）**：用于计算优势函数（advantage）

训练过程采用**双层优化（Bi-level Optimization）**：
- **内循环（Inner Loop）**：更新Agent的参数
- **外循环（Outer Loop）**：更新Meta-Network的参数

## 2. meta_train.ipynb中的训练流程

### 2.1 初始化阶段

```python
# 创建多个智能体（Population-based training）
num_agents = 2
num_steps = 800
rollout_len = 16
num_inner_steps = 2
batch_size_per_device = 32

# 初始化元网络参数
update_rule_params = random_update_rule_params  # 或使用disco_103_params
meta_opt = optax.adam(learning_rate=5e-4)
meta_opt_state = meta_opt.init(update_rule_params)

# 创建多个智能体实例
agents = []
agents_states = []
for rng_key in jax.random.split(rng_key, num_agents):
    agents.append(MetaTrainAgent(...))
    agents_states.append(agents[-1].init_state(rng_key))
```

每个智能体状态包含：
- `learner_state`：Agent的参数和优化器状态
- `actor_state`：Actor的LSTM隐藏状态
- `value_state`：价值函数的参数
- `env_state`和`env_timestep`：环境状态

### 2.2 元训练主循环

```python
for meta_step in tqdm.tqdm(range(num_steps)):
    (
        step_update_rule_params,      # 更新后的元网络参数
        step_meta_opt_state,          # 元优化器状态
        step_agents_states,           # 所有智能体状态
        metrics[meta_step],           # 训练指标
        meta_log[meta_step],          # 元学习日志
    ) = jitted_meta_update(
        update_rule_params=step_update_rule_params,
        meta_opt_state=step_meta_opt_state,
        agents_states=step_agents_states,
        rng=step_rngs,
    )
```

## 3. 内循环：Agent参数更新

### 3.1 数据收集阶段

```python
# 为每个智能体生成训练轨迹
train_rollouts = [None] * num_agents
valid_rollouts = [None] * num_agents

for agent_i in range(num_agents):
    agent, state = agents[agent_i], agents_states[agent_i]
    rollouts = [None] * num_inner_steps

    # 收集num_inner_steps条轨迹，每条长度为rollout_len
    for step_i in range(num_inner_steps):
        state, rollouts[step_i] = agent.unroll_actor(
            state, rngs_per_step[step_i], rollout_len
        )
    train_rollouts[agent_i] = utils.tree_stack(rollouts)

    # 生成验证轨迹（2倍长度）
    agents_states[agent_i], valid_rollouts[agent_i] = agent.unroll_actor(
        state, rngs_per_agent_act[agent_i], 2 * rollout_len
    )
```

### 3.2 Agent参数更新步骤

在`_inner_step`函数中执行：

```python
def _inner_step(carry, inputs):
    """使用当前更新规则更新Agent和价值函数的参数"""
    update_rule_params, learner_state, actor_state, value_state = carry
    actor_rollout, learner_rng = inputs

    # 步骤1：更新Agent的参数
    new_learner_state, new_actor_state, metrics = agent.learner_step(
        rng=learner_rng,
        rollout=actor_rollout,
        learner_state=learner_state,
        agent_net_state=actor_state,
        update_rule_params=update_rule_params,  # 使用元网络参数
        is_meta_training=True,
    )

    # 步骤2：更新价值函数
    agent_out, _ = agent.unroll_net(
        learner_state.params, actor_state, actor_rollout
    )
    new_value_state, _, _ = agent.value_fn.update(
        value_state, actor_rollout, agent_out['logits']
    )

    return (
        update_rule_params,
        new_learner_state,
        new_actor_state,
        new_value_state,
    ), metrics
```

**关键点**：
- `agent.learner_step`中使用`update_rule_params`（元网络参数）来生成训练目标
- 元网络会生成三个目标：π̂（策略目标）、ŷ（状态预测目标）、ẑ（状态-动作预测目标）
- Agent通过最小化与这些目标的KL散度来更新参数

### 3.3 扫描多个内循环步骤

```python
# 执行N个内循环步骤（通过JAX的scan实现高效计算）
learner_rngs = jax.random.split(train_rng, unroll_len)
(_, new_learner_state, new_actor_state, new_value_state), train_metrics = (
    jax.lax.scan(
        _inner_step,
        (
            update_rule_params,
            agent_state.learner_state,
            agent_state.actor_state,
            agent_state.value_state,
        ),
        (train_rollouts, learner_rngs),
    )
)
```

## 4. 外循环：Meta-Network参数更新

### 4.1 计算元损失函数

在`_outer_loss`函数中：

```python
def _outer_loss(
    update_rule_params: types.MetaParams,
    agent_state: MetaTrainState,
    train_rollouts: types.ActorRollout,
    valid_rollout: types.ActorRollout,
    rng: chex.PRNGKey,
):
    """计算更新规则的元损失"""

    # 1. 执行N个内循环步骤（Agent参数更新）
    (_, new_learner_state, new_actor_state, new_value_state), train_metrics = (
        jax.lax.scan(_inner_step, ...)
    )

    # 2. 在验证轨迹上运行推理
    agent_rollout_on_valid, _ = hk.BatchApply(
        lambda ts: agent.actor_step(
            actor_params=new_learner_state.params,
            rng=valid_rng,
            timestep=ts,
            actor_state=valid_rollout.first_state(time_axis=0),
        )
    )(valid_rollout.to_env_timestep())

    # 3. 计算验证轨迹上的价值函数
    value_out, _, _, _ = agent.value_fn.get_value_outs(
        new_value_state, valid_rollout, agent_rollout_on_valid['logits']
    )

    # 4. 计算策略梯度损失
    actions_on_valid = valid_rollout.actions[:-1]
    logits_on_valid = agent_rollout_on_valid['logits'][:-1]
    adv_t = jax.lax.stop_gradient(value_out.normalized_adv)

    pg_loss_per_step = utils.differentiable_policy_gradient_loss(
        logits_on_valid, actions_on_valid, adv_t=adv_t, backprop=False
    )

    # 5. 计算正则化损失
    reg_loss = 0

    # 5.1 熵正则化（鼓励探索）
    reg_loss += -1e-2 * distrax.Softmax(logits_on_valid).entropy().mean()

    # 5.2 验证集的熵正则化
    agent_out_on_valid = agent_rollout_on_valid.agent_outs
    z_a = utils.batch_lookup(agent_out_on_valid['z'][:-1], actions_on_valid)
    y_entropy_loss = -jnp.mean(distrax.Softmax(agent_out_on_valid['y']).entropy())
    z_entropy_loss = -jnp.mean(distrax.Softmax(z_a).entropy())
    reg_loss += 1e-3 * (y_entropy_loss + z_entropy_loss)

    # 5.3 训练集的正则化（减少目标偏差）
    dp, dy, dz = train_meta_out['pi'], train_meta_out['y'], train_meta_out['z']
    reg_loss += 1e-3 * jnp.mean(jnp.square(jnp.mean(dy, axis=(1, 2, 3))))
    reg_loss += 1e-3 * jnp.mean(jnp.square(jnp.mean(dz, axis=(1, 2, 3))))
    reg_loss += 1e-3 * jnp.mean(jnp.square(jnp.mean(dp, axis=(1, 2, 3))))

    # 5.4 目标KL散度损失（保持目标稳定性）
    logits = train_meta_out['target_out']['logits'][:, :-1]
    target_kl_loss = rlax.categorical_kl_divergence(
        jax.lax.stop_gradient(logits), dp
    )
    reg_loss += 1e-2 * jnp.mean(target_kl_loss)

    # 6. 最终元损失
    meta_loss = pg_loss_per_step.mean() + reg_loss

    return meta_loss, (new_agent_state, train_metrics, meta_log)
```

**元损失组成**：
1. **策略梯度损失**：衡量Agent在验证集上的表现
2. **熵正则化**：鼓励探索
3. **目标稳定性正则化**：防止元网络生成的目标过于极端
4. **KL散度正则化**：保持目标与当前策略的接近度

### 4.2 计算元梯度

```python
def calculate_meta_gradient(
    update_rule_params: types.MetaParams,
    agent_state: MetaTrainState,
    train_rollouts: types.ActorRollout,
    valid_rollout: types.ActorRollout,
    rng: chex.PRNGKey,
    agent: MetaTrainAgent,
    axis_name: str | None = axis_name,
):
    """计算单个智能体的元梯度"""

    # 使用JAX的自动微分计算元梯度
    meta_grads, outputs = jax.grad(_outer_loss, has_aux=True)(
        update_rule_params,      # 对这个参数求梯度
        agent_state,
        train_rollouts,
        valid_rollout,
        rng
    )

    new_agent_state, train_metrics, meta_log = outputs

    # 跨设备平均梯度
    if axis_name is not None:
        (meta_grads, train_metrics, meta_log) = jax.lax.pmean(
            (meta_grads, train_metrics, meta_log), axis_name
        )

    return meta_grads, (new_agent_state, train_metrics, meta_log)
```

**关键点**：
- 元梯度通过整个内循环的反向传播计算得到
- 这是DiscoRL的核心：通过验证集上的表现来优化更新规则本身

### 4.3 聚合多个智能体的梯度并更新

```python
def meta_update(
    update_rule_params: types.MetaParams,
    meta_opt_state: optax.OptState,
    agents_states: list[MetaTrainState],
    rng: chex.PRNGKey,
    axis_name: str | None = axis_name,
):
    """计算元参数的更新"""

    # 1. 为每个智能体计算元梯度
    meta_grads = [None] * num_agents
    for agent_i in range(num_agents):
        meta_grads[agent_i], (agents_states[agent_i], metrics, meta_log) = (
            calculate_meta_gradient(
                update_rule_params=update_rule_params,
                agent_state=agents_states[agent_i],
                train_rollouts=train_rollouts[agent_i],
                valid_rollout=valid_rollouts[agent_i],
                rng=rngs_per_agent_upd[agent_i],
                agent=agents[agent_i],
                axis_name=axis_name,
            )
        )

    # 2. 平均所有智能体的元梯度
    avg_meta_gradient = jax.tree.map(
        lambda x: x.mean(axis=0), utils.tree_stack(meta_grads)
    )

    # 3. 通过元优化器更新元参数
    meta_update, meta_opt_state = meta_opt.update(
        avg_meta_gradient, meta_opt_state
    )
    update_rule_params = optax.apply_updates(update_rule_params, meta_update)

    # 4. 记录日志
    meta_log['meta_grad_norm'] = optax.global_norm(avg_meta_gradient)
    meta_log['meta_up_norm'] = optax.global_norm(meta_update)
    meta_log['rewards'] = utils.tree_stack(rewards).mean()

    return update_rule_params, meta_opt_state, agents_states, metrics, meta_log
```

## 5. DiscoRL的核心创新：自举机制（Bootstrapping）

DiscoRL发现的关键机制是使用**未来的预测来构建当前的训练目标**：

```python
# 在disco.py的unroll_meta_net方法中
# 元网络接收：
# - 当前状态的特征
# - 当前预测 y(s), z(s,a)
# - 未来预测 y(s'), z(s',a')
# - 奖励信息

# 生成目标：
# π̂_t = f_π(y_t, z_t, y_{t+1}, z_{t+1}, r_t, ...)
# ŷ_t = f_y(y_t, z_t, y_{t+1}, z_{t+1}, r_t, ...)
# ẑ_t = f_z(y_t, z_t, y_{t+1}, z_{t+1}, r_t, ...)
```

这种自举机制允许算法：
1. 利用未来信息来改进当前的学习目标
2. 自动发现类似时序差分（TD）学习的机制
3. 比传统的固定更新规则更加灵活

## 6. 完整训练循环总结

```
for meta_step in range(num_steps):
    # ========== 阶段1：数据收集 ==========
    for agent_i in range(num_agents):
        # 收集训练轨迹
        for inner_step in range(num_inner_steps):
            train_rollouts[agent_i][inner_step] = agent.unroll_actor(rollout_len)
        # 收集验证轨迹
        valid_rollouts[agent_i] = agent.unroll_actor(2 * rollout_len)

    # ========== 阶段2：内循环（Agent更新） ==========
    for agent_i in range(num_agents):
        for inner_step in range(num_inner_steps):
            # 使用元网络生成目标
            targets = meta_network(update_rule_params, rollout)
            # 更新Agent参数
            agent_params = agent_optimizer.update(agent_params, targets)

    # ========== 阶段3：外循环（Meta-Network更新） ==========
    for agent_i in range(num_agents):
        # 在验证集上评估Agent表现
        validation_loss = evaluate_on_validation(agent_params, valid_rollout)
        # 通过整个内循环反向传播计算元梯度
        meta_grads[agent_i] = grad(validation_loss, update_rule_params)

    # 平均所有智能体的梯度
    avg_meta_grad = mean(meta_grads)
    # 更新元网络参数
    update_rule_params = meta_optimizer.update(update_rule_params, avg_meta_grad)
```

## 7. 关键代码文件说明

### 7.1 agent.py中的learner_step

```python
def learner_step(
    self,
    rng: chex.PRNGKey,
    rollout: types.ActorRollout,
    learner_state: LearnerState,
    agent_net_state: types.HaikuState,
    update_rule_params: Optional[types.MetaParams] = None,
    is_meta_training: bool = False,
) -> tuple[LearnerState, types.HaikuState, dict[str, chex.Array]]:
    """执行一步learner更新"""

    # 1. 通过网络前向传播
    agent_out, new_agent_net_state = self.unroll_net(
        learner_state.params, agent_net_state, rollout
    )

    # 2. 使用更新规则计算梯度和更新
    updates, new_opt_state, metrics = self.update_rule.get_update(
        rng=rng,
        params=learner_state.params,
        opt_state=learner_state.opt_state,
        rollout=rollout,
        agent_out=agent_out,
        update_rule_params=update_rule_params,  # 元网络参数
        is_meta_training=is_meta_training,
    )

    # 3. 应用更新
    new_params = optax.apply_updates(learner_state.params, updates)

    return (
        LearnerState(params=new_params, opt_state=new_opt_state),
        new_agent_net_state,
        metrics,
    )
```

### 7.2 disco.py中的get_update

```python
def get_update(
    self,
    rng: chex.PRNGKey,
    params: types.Params,
    opt_state: types.OptState,
    rollout: types.ActorRollout,
    agent_out: dict[str, chex.Array],
    update_rule_params: Optional[types.MetaParams] = None,
    is_meta_training: bool = False,
):
    """使用元网络生成的目标计算参数更新"""

    # 1. 展开元网络得到训练目标
    targets, meta_state, meta_out = self.unroll_meta_net(
        rng=rng,
        update_rule_params=update_rule_params,
        rollout=rollout,
        agent_out=agent_out,
        prev_meta_state=opt_state.meta_state,
    )
    # targets包含: π̂, ŷ, ẑ

    # 2. 计算损失（KL散度）
    loss_fn = lambda p: self._compute_loss(
        p, rollout, agent_out, targets
    )

    # 3. 计算梯度
    grads = jax.grad(loss_fn)(params)

    # 4. 应用梯度裁剪
    updates, new_base_opt_state = self.base_optimizer.update(
        grads, opt_state.base_opt_state, params
    )

    return updates, new_opt_state, metrics
```

## 8. 超参数和训练配置

```python
# 元训练配置
num_steps = 800                    # 元训练步数
num_agents = 2                     # 智能体数量（population size）
rollout_len = 16                   # 单次rollout长度
num_inner_steps = 2                # 内循环步数
batch_size_per_device = 32         # 每个设备的批次大小

# Agent网络配置
agent_settings.net_settings.name = 'mlp'
agent_settings.net_settings.net_args = dict(
    dense=(512, 512),              # MLP隐藏层
    model_arch_name='lstm',        # 使用LSTM
    head_w_init_std=1e-2,
    model_kwargs=dict(
        head_mlp_hiddens=(256,),
        lstm_size=256,
    ),
)
agent_settings.learning_rate = 5e-4

# 价值函数配置
value_fn_config = types.ValueFnConfig(
    net='mlp',
    net_args=dict(dense=(256, 256), ...),
    learning_rate=1e-3,
    discount_factor=0.99,          # γ
    td_lambda=0.96,                # λ (for GAE)
    outer_value_cost=1.0,
)

# 元优化器
meta_opt = optax.adam(learning_rate=5e-4)
```

## 9. 并行化和设备分布

```python
# 使用JAX的pmap进行多设备并行
devices = jax.devices()
jitted_meta_update = jax.pmap(meta_update, axis_name=axis_name, devices=devices)

# 将参数复制到所有设备
step_update_rule_params = jax.device_put_replicated(update_rule_params, devices)
step_meta_opt_state = jax.device_put_replicated(meta_opt_state, devices)
step_agents_states = jax.device_put_replicated(agents_states, devices)

# 执行并行元更新
for meta_step in range(num_steps):
    step_rngs = jax.random.split(step_rngs, len(devices))
    (
        step_update_rule_params,
        step_meta_opt_state,
        step_agents_states,
        metrics[meta_step],
        meta_log[meta_step],
    ) = jitted_meta_update(...)
```

## 10. 训练监控指标

```python
# 元学习日志
meta_log = {
    'adv': ...,                    # 平均优势函数值
    'normalized_adv': ...,         # 归一化优势函数值
    'entropy': ...,                # 策略熵
    'value': ...,                  # 价值函数估计
    'meta_loss': ...,              # 元损失
    'pg_loss': ...,                # 策略梯度损失
    'reg_loss': ...,               # 正则化损失
    'meta_grad_norm': ...,         # 元梯度范数
    'meta_up_norm': ...,           # 元更新范数
    'rewards': ...,                # 平均奖励
    'pos_rewards': ...,            # 正奖励数量
    'neg_rewards': ...,            # 负奖励数量
}
```

## 11. 总结

DiscoRL通过meta_train.ipynb实现的训练机制具有以下特点：

1. **双层优化结构**：
   - 内循环快速适应具体任务
   - 外循环优化通用的更新规则

2. **基于种群的训练**：
   - 多个智能体并行训练
   - 提高发现鲁棒算法的可能性

3. **端到端可微**：
   - 通过JAX实现整个训练过程的自动微分
   - 元梯度通过内循环反向传播计算

4. **自举机制**：
   - 元网络利用未来预测改进当前目标
   - 自动发现类似TD学习的机制

5. **灵活的目标生成**：
   - 元网络根据状态动态生成训练目标
   - 不同于固定的更新公式

这种设计使DiscoRL能够自动发现性能优于人工设计算法的强化学习更新规则。
