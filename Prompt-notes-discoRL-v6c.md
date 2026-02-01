
read file @Prompt-notes-discoRL-v6c.md and complete the all tasks.


# Role
-------------------
You are a senior AI algorithm researcher, proficient in Neural Networks, DL, RL, LSTM, Python, PyTorch, JAX, AlphZero, MuZero, DiscoRL, and related knowledge and algorithms.


# Task
-------------------
1. First, read all the documents under the directory 'discoRL_Doc/', and read all the documents and code under the directory 'disco_rl/'.

2. Then, create an outline of the DiscoRL algorithm. Write the outline into a Pseudocode file named 'DiscoRL-pseudocode-v6c.py'.

3. Answer each question in 'Questions' one by one, and write the answers into a markdown file named 'DiscoRL-notes-v6c.md'. Answer the questions in English.
When answering questions, first provide a textual response, then list the corresponding source code to explain the problem. Provide comments on key operations in the code to help me better understand its execution logic and the rationale behind its design.

4. When explaining code, specify which file and line number it comes from. The line number should be displayed in the original file.


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

