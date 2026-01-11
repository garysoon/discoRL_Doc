

# Role
-------------------
You are a senior AI algorithm researcher, proficient in Python, PyTorch, JAX, Neural Networks, LSTM, AlphZero, MuZero, DiscoRL, and related knowledge and algorithms.


# Task
-------------------
1. First, read all the documents under the directory 'discoRL_Doc/', and read all the code under the directory 'disco_rl/'.

2. Then, create an outline of the DiscoRL algorithm. Write the outline into a Pseudocode file named 'DiscoRL-pseudocode-v1.py'.

3. Answer each question in 'Questions' one by one, and write the answers into a markdown file named 'DiscoRL-notes-ClaudeCode-v3.md'. Answer the questions in English.

4. The Opus 4.5 model was used throughout. Think mode was enabled before each task.

5. When explaining code, specify which file and which lines it is from.


# Questions
-------------------
1. what is the neural network architecture of the 'Agent-network'?

2. how to train the Agent-network ?

3. how to product the predictions (π, y, z) from Agent-network ?
   what are the inputs for Agent producing predictions ?
   what are the ouputs of the predicitons from Agent ?
   how to transform the observations from enviroments to inputs ?

4. how to calculate loss between Agent rollouts and targets from Meta-network ?

5. how to calculate gradients of Agent-network ?

6. how to update the parameters of Agent-network ?

7. What is the neural network architecture of the 'Meta-network'?

8. how to train the Meta-network ?

9. how to product the targets (π_hat, y_hat, z_hat) from Meta-network ?
   what are the inputs for Meta-network producing targets ?
   what are the ouputs of the targets from Meta-network ?

10. how to calculate loss of meta-network ?

11. how to calculate gradient of meta-network ?

12. how to update the parameters of Meta-network ?

13. how to calculate 'Advantage estimates' ?

14. What is the purpose of 'Advantage estimates'? Where are they used? When are they used?

15. What is the function of the 'Value Function' in file value_fn.py? Why is a value function necessary? Why does the value function change? When the value function changes, what exactly is being modified?




