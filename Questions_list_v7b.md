
# Questions 
-------------------

1. what is the neural network architecture of the 'Agent-network'? Where is the Neural Network of the 'Agent-network' defined? How is it defined?

2. how to train the Agent-network ?

3. how to generate the predictions (π, y, z) from Agent-network ?
   what are the inputs for Agent producing predictions ?
   what are the ouputs of the predicitons from Agent ?
   how to transform the observations from enviroments to inputs ?

4. how to calculate loss between Agent rollouts and targets from Meta-network ?

5. how to calculate gradients of Agent-network ?

6. how to update the parameters of Agent-network ?

7. What is the neural network architecture of the 'Meta-network'? Where is the Neural Network of the 'Meta-network' defined? How is it defined?

8. how to train a new Meta-network? 

9. how to finetune a Meta-network? Let's start with the explanation in the file @disco_rl/colabs/meta_train.ipynb.

10. how to generate the targets (π_hat, y_hat, z_hat) from Meta-network ?
   what are the inputs for Meta-network producing targets ?
   what are the ouputs of the targets from Meta-network ?

11. how to calculate loss of meta-network ?

12. how to calculate gradient of meta-network ?

13. how to update the parameters of Meta-network ?

14. how to calculate 'Advantage estimates' ?

15. What is the purpose of 'Advantage estimates'? Where are they used? When are they used? how are ‘Advantage estimates’ calculated? Where are they calculated? What are the Inputs and Outputs respectively?

16. What is the function of the 'Value Function' in file value_fn.py? Why is a value function necessary? Why does the value function change? When the value function changes, what exactly is being modified?

17. how to evaluate a meta-network? Let's start by explaining from the file @disco_rl/colabs/eval.ipynb.


18. In the discoRL algorithm, how are the following concepts and components assembled together? How do they cooperate and interact with each other? How does data flow between them?
   Agent, 
   Meta-network, 
   ValueFunction, 
   Advantage, V-tarce, Retrace, 
   TD-error, 
   Agent-LSTM, 
   Meta-LSTM, 
   Meta-RNN，
   inner-loop,
   outer-loop, 
   Actor-Critic, 
   Q_value, aux_pi, (π, y, z),
   env-model,
   action-model, 


19. The three neural networks: 1) agent, 2) meta-network, and 3) value function. How do the three neural networks interact and cooperate? How does data flow between them?


20. What is the relationship between EMA and Normalize? Are they equivalent? When should EMA be used, and when should normalize be used?


21. How are ‘Advantage estimates’ calculated? Where are they calculated? What are the Inputs and Outputs respectively?


22. What do Current, Behavior, and Target policies refer to respectively?


23. Should the shape of z in the agent's rollouts be [T, B, 600] or [T, B, 600, A]?


24. Why do many calculations require `learner_state`, `actor_state`, and `meta_state` as input? Is it because the LSTM network needs state?


25. Why do the return values for agent predictions and meta-targets still contain various states? Is it because the next calculation needs these states as input to the LSTM network?


26. Why is q-loss missing from agent loss? Agent loss includes π-loss, y-loss, z-loss, and aux_pi-loss, but why is q-loss absent?


27. What is the difference between aux_pi_a and aux_pi? What is the difference between q_a and q? What operation does the batch_lookup() function perform? What is its function?


28. What are the roles of Q-values and State-values?
What is Value-net? How is it defined? What is Q-net? How is it defined?
Value-net uses the V-trace algorithm to generate values. How does the V-trace algorithm work? What are the inputs and outputs?
Q-net uses the Retrace algorithm to generate Q-values. How does the Retrace algorithm work? What are the inputs and outputs?
Why are Q-values and state-values interchangeable? When a Q-value does not exist, a state-value is used instead. Why is this?
What are the differences and relationships between the following four types of values?
   - State values, V-trace, value_net_out.
   - Q-values, Retrace, q_net_out.
   - target_value_net_out.
   - target_q_net_out.


29. Please explain whether the following understanding is correct:
Value target = V(s) + TD_error
TD error     = target - prediction
Advantage    = V-trace_return - V(s)


30. Could it be understood as follows: The state-value is a predicted scalar value given by the agent based on the current state, representing the estimated game score.
The Q-value, on the other hand, is the agent's prediction of rewards for the next 10 or 20 moves. The sum of these predicted future rewards represents the score the agent, based on its current capabilities, is likely to achieve in the future.
target_value_net_out and target_q_net_out are the meta-network's predicted scores for the current state and the sum of predicted future rewards, respectively.


31. What does "importance weights" mean? See value_utils.py:495-509
What does rho mean? What is mu in the formula?
rho = pi(a|s) / mu(a|s) = exp(log_pi - log_mu)


32. What is the purpose of normalized_adv?
Advantage Normalization (value_utils.py:203-222)
normalized_adv = (adv - mean) / std


33. How are ‘Advantage estimates’ calculated? Where are they calculated? What are the Inputs and Outputs?
How are the average action results or game score calculated?


34. How is the 'Value Function' modified? Why is the 'Value Function' continuously modified?


35. What is the role of the actor-critical algorithm in the DiscoRL algorithm? In which stage does it operate? What part of the data does it process? What role does it play?


36. In agent-loss, there are components of (π, y, z, aux_pi), but why is there no component of q?
Answer: q_loss is calculated in disco.py:agent_loss_no_meta().
Question: What does no_meta mean in agent_loss_no_meta()?


37. Meta-network targets: stopped or not stopped? What does it mean if meta-network targets are stopped? What does it mean if they are not stopped? What does it mean when backprop=True, gradients flow through targets?


38. How does the meta-network use predictions to update the agent’s policy in practice, and what exactly serves as the baseline for variance reduction? Specifically, do these predictions refer to agent-generated predictions or meta-network targets, and through what mechanism does the meta-network influence the agent policy update process? Additionally, how does this baseline differ from or relate to the conventional value function baseline, and how can the apparent inconsistencies between the paper and the implementation be explained?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Page 314, chapter 'Meta-network'，Fourth point: use predictions to update the policy (for example, to provide a baseline for variance reduction).


39. How is the advantage computed using A2C (or V-trace)? How is the meta-value function trained and used? What is the relationship between A2C, advantage, ValueFunction (in code), and ∇J(θ)? Is the meta-value function the same as the code's ValueFunction? Is it a neural network or a plain function? Provide the exact formula for advantage and how it is used in meta-gradient computation.
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, 'Meta-optimization' chapter.


40. What is the precise relationship between the primary policy π and the auxiliary policy prediction p (aux_pi)? Why is aux_pi introduced despite the existence of π, and do they contain redundant information or complementary information? If they are not redundant, what information is missing from π that aux_pi provides? Furthermore, why do introducing auxiliary predictions such as q(s,a) and p(s,a) facilitate the discovery of new concepts?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Discovery method'.


41. Why is aux_pi (p) not included in the agent output vector fθ = [π, y, z, q], and how is aux_pi actually computed and generated within the overall architecture?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-network chapter, f_θ = [π, y, z, q].


42. When the meta-objective is modified to include regularization terms (e.g., J(η) = E[J(θ) − L_ent(θ) − L_kl(θ)]), how is the meta-gradient ∇J(η) formally derived? How does this relate to the original formulation ∇J(η) = ∇θ · ∇J(θ)? Additionally, how are the agent gradient, ∇θ, and the objective J(θ) defined and computed step by step from trajectories, and what is the precise relationship between J(η) and J(θ)?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-optimization stabilization chapter.


43. What does “sharing weights across action dimensions” mean in the action-invariant architecture, and how does it enable a single meta-network to handle environments with different action space sizes? In particular, how are action-dependent quantities like π, q(s,a), and p(s,a) represented under this scheme, and what determines the dimensionality of π when environments have different numbers of actions?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Meta-network'.


44. Why are targets constructed using future predictions and policies (i.e., “the future determines the present”), instead of using current predictions or directly training on trajectories such as (π, y, z, rewards, terminal)? What is the motivation for introducing the intermediate prediction → target → prediction structure, rather than learning directly from simulated or observed trajectories?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Meta-network'.


45. How are the targets q̂ and p̂ computed in detail, including the use of Retrace, two-hot projection, and target networks? What roles do μ(s,a), r(s,a), and other inputs play in this computation? Are q̂ and p̂ directly produced by the meta-network, or computed via separate procedures? Additionally, how should q(s,a) be interpreted relative to traditional value functions (e.g., v or expected return), and how does introducing q(s,a) help avoid rediscovering action-value functions?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Agent optimization'; Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Table 1.


46. What are the respective roles of the core LSTM and the meta-RNN in generating targets within the meta-network? What outputs does the meta-RNN produce, and how is its internal state used (e.g., as input to the decoder φ_dec for generating π̂)? Furthermore, is the meta-RNN strictly necessary, or could it be removed or simplified, and if so, which component would take over its functionality?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-network chapter; Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 2.2.


47. How is the auxiliary policy prediction (aux_pi or p) in the agent's “policy and predictions” outputs computed, and by which module? In the Agent optimization chapter, p^ = π(s') where s' is the next state—meaning the target in targets equals the next state’s π. 
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, 'Agent optimization' chapter.


48. Is it feasible to further eliminate pre-defined concepts like $\pi, q,$ and $aux\_pi$ to provide the discovery process with greater autonomy?
What specific data types constitute the "trajectory" input for the meta-network?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Meta-network'.


49. Why does the implementation of $aux\_pi$ differ between the paper ($p(s, a)$) and the code ($p(a, a)$), and what are the implications of this change? Does this imply a change in semantic meaning or vector dimensionality?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Discovery method'.


50. What is the definition of the sliding window mechanism used in meta-gradient estimation?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Meta-optimization'.


51. What does the Interquartile Mean (IQM) signify in the context of performance evaluation?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Empirical result'.


52. What is the formal relationship between the agent's action-value $q(s, a)$ and the 'ValueFunction' module? Do they represent the same underlying concept?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, page 314, chapter 'Effect of discovering new predictions'.


53. How can the capabilities of DiscoRL103 be distilled into a model using a Large Language Model (LLM) as the meta-network backbone?


54. Which architecture, Transformer or LSTM, demonstrates superior performance in the context of the DiscoRL meta-network?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-network chapter.


55. What is the definition and role of "intermediate embeddings" in the meta-network?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-network chapter.


56. Is the uniform averaging of agent gradients the optimal approach, or are there more effective population aggregation methods?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-optimization stabilization chapter.


57. What is the typical range and significance of the prediction entropy $H(y(s))$?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-optimization stabilization chapter.


58. How is the target network policy π_θ' defined and calculated using the exponential moving average of parameters?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-optimization stabilization chapter.


59. What is the function of MixFlow-MG in the computational framework?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Implementation details chapter.


60. What is the definition and technical purpose of gradient clipping in meta-optimization?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Implementation details chapter.


61. What are the differences between a meta-step and an environment time-step, and how are trajectories structured within a batch?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Implementation details chapter.


62. Does the described convolutional and LSTM-based architecture refer to the meta-network, the agent-network, or a specific evaluation model?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Hyperparameters and evaluation chapter.


63. Why are different numbers of random seeds utilized for different benchmarks like Atari, Crafter, and Sokoban?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Hyperparameters and evaluation chapter.


64. What is the role of the Multi-Layer Perceptrons (MLPs) used in the analysis details, and how do they facilitate the interpretation of agent predictions?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Analysis details chapter.


65. What primary conclusions were drawn from the regression and classification analysis presented in Extended Data Fig. 2?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Analysis details chapter.


66. What does the hyperparameter "sweep" analysis in Extended Data Fig. 1c indicate regarding the robustness of DiscoRL?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Extended Data Fig. 1.


67. In Extended Data Fig. 2, does $q(a)$ represent the advantage of the action-value function, and how is it normalized?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Extended Data Fig. 2.


68. What is the definition and role of the behavior policy $\mu$ in off-policy corrections?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Table 1.


69. Does 'episode termination' refer to the terminal signal of an individual game session?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Table 1.


70. What are the 'other agent outputs' that the meta-network receives as input?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 1.1.


71. In the context of DiscoRL, is the 'objective function' equivalent to the meta-objective $J(\eta)$?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 1.1.


72. What is meant by the statement that predictions may "not be conditional on multi-action sequences"?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 1.1.


73. Why is DiscoRL classified as an off-policy algorithm, and can it be adapted for purely on-policy learning?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 1.1.


74. What is the source and derivation of the policy gradient formulation ∇L = Q * ∇logπ ?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), section 1.2.


75. How is the target policy $\hat{\pi}$ computed using the action-invariant encoder $\phi_{enc}$ and decoder $\phi_{dec}$ within the meta-network?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Figure 4, section 2.1.


76. Are the action-value function $q(s, a)$ and the auxiliary policy prediction $p(s, a)$ (referred to as $aux\_pi$) generated directly as outputs of the agent network, or are they produced by a separate architectural module?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Page 314, chapter 'Agent network'.


77. What is the formal interpretation of the statement "$\theta'$ (or $\theta^{-}$) is an exponential moving average of parameters $\theta$," and how does this parameter-level smoothing function within the algorithm's logic?
[Reference] Discovering state-of-the-art reinforcement learning algorithms, Methods -> Meta-network section.


78. Does the exponential moving average (EMA) denoted by θ' (or θ-) apply to the agent's output vectors (π, y, z) or specifically to the underlying neural network weights and parameters θ?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Table 1: Meta-network inputs.


79. In the context of the stated limitations, how is a "learning model" defined (e.g., as a transition or world model), and why does the current architecture preclude its acquisition from observations?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Limitations section.


80. What is the mathematical definition of a "mixture of Gaussians," and in what capacity is it used to generate synthetic data for approximating distributional value targets?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), Fig. 1: Approximation of C51 and PPO with meta-network.


81. Within the provided open-source implementation, which specific neural network module corresponds to the action-invariant encoder φ_enc described in the paper?
[Reference] Discovering state-of-the-art reinforcement learning algorithms (Supplementary information), 2.1 Action-invariant architecture section.


82. Gradient Field Analysis: 
How does the target-based meta-learning framework specifically circumvent the "conservative vector field" limitation of scalar loss functions, and what are the empirical benefits of learning non-conservative semi-gradients in complex environments?


83. Zero-Shot Generalization: 
The papers claim DiscoRL generalizes to unseen environments like ProcGen and NetHack. To what extent does the exclusion of raw observations from the meta-network's input contribute to this generalization, and would adding compressed state representations (e.g., from a VAE) improve or degrade this robustness?


84. Bootstrap Horizon and Credit Assignment: 
Analysis shows that DiscoRL uses a bootstrapping mechanism (future predictions inform current targets). How does the meta-learned bootstrapping horizon adapt to environments with varying reward sparsity, and is there evidence of the rule performing multi-step credit assignment more effectively than manual TD($\lambda$)?


85. Meta-RNN Dynamics: 
The meta-RNN processes information across agent lifetimes. Does this allow DiscoRL to effectively "discover" meta-RL strategies (like RL$^2$) where the agent's internal state acts as a memory of past trials within a single lifetime?


86. Inductive Biases vs. Expressivity: 
While DiscoRL is more expressive than prior work, it still assumes a specific functional form ($\pi, y, z$). What are the theoretical limits of this inductive bias, and could a more primitive interface (e.g., directly outputting parameter updates $\Delta \theta$) potentially outperform DiscoRL?

