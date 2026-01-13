
read file @Prompt-ClaudeCode-v4.md and complete the all tasks.



# Role
-------------------
You are a senior AI algorithm researcher, proficient in Neural Networks, DL, RL, LSTM, Python, PyTorch, JAX, AlphZero, MuZero, DiscoRL, and related knowledge and algorithms.


# Task
-------------------
1. First, read all the documents under the directory 'discoRL_Doc/', and read all the documents and code under the directory 'disco_rl/'.

2. Then, create an outline of the DiscoRL algorithm. Write the outline into a Pseudocode file named 'DiscoRL-pseudocode-v3.py'.

3. Answer each question in 'Questions' one by one, and write the answers into a markdown file named 'DiscoRL-notes-ClaudeCode-v5.md'. Answer the questions in English.

4. The Opus 4.5 model was used throughout. Think mode was enabled before each task.

5. When explaining code, specify which file and line number it comes from. The line number should be displayed in the original file, not in the merged file.


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

15. What is the purpose of 'Advantage estimates'? Where are they used? When are they used?

16. What is the function of the 'Value Function' in file value_fn.py? Why is a value function necessary? Why does the value function change? When the value function changes, what exactly is being modified?

17. how to evaluate a meta-network? Let's start by explaining from the file @disco_rl/colabs/eval.ipynb.

