# RLControl
Implementation of Continuous Control RL Algorithms. 

Repository used for our paper [Actor-Expert: A Framework for using Q-learning in Continuous Action Spaces](https://arxiv.org/abs/1810.09103).

webpage: https://sites.google.com/ualberta.ca/actorexpert

## Available Algorithms
* Q-learning methods
  * Actor-Expert, Actor-Expert+: [Actor-Expert: A Framework for using Q-learning in Continuous Action Spaces](https://arxiv.org/abs/1810.09103)
  * Actor-Expert with PICNN
  * Wire-Fitting: [Reinforcement Learning with High-dimensional, Continuous Actions](http://www.leemon.com/papers/1993bk3.pdf) 
  * Normalized Advantage Functions(NAF): [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)
  * (Partial) Input Convex Neural Networks(PICNN): [Input Convex Neural Networks](https://arxiv.org/abs/1609.07152) - adapted from github.com/locuslab/icnn
  * QT-Opt: [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293) - both single and mixture gaussian
  
* Policy Gradient methods
  * Deep Deterministic Policy Gradient(DDPG): [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
  * Advantage Actor-Critic baseline with Replay Buffer: Not to be confused with ACER


## Installation
Create virtual environment and install necessary packages through "pip3 -r requirements.txt"

## Usage
Settings for available environments and agents are provided in `jsonfiles/` directory

**Example:**

ENV=Pendulum-v0 (must match jsonfiles/environment/*.json name)

AGENT=ddpg (must match jsonfiles/agent/*.json name)

INDEX=0 (useful for running sweeps over different settings and doing multiple runs)
INDEX is the index into the settings and run to use. The agent JSON file defines
the number of settings. Each specific combination of variables in the JSON file
represents the settings number. For example, if we had two learning rates and
we wanted to perform sweeps over them, then in the agent JSON file
we would have `lr1: [0.01, 0.001]` and `lr2: [0.1, 0.01]`. Then:
```
INDEX=0 <==> run 1 of lr1 = 0.01 lr2 = 0.1
INDEX=1 <==> run 1 of lr1 = 0.01 lr2 = 0.01
INDEX=2 <==> run 1 of lr1 = 0.001 lr2 = 0.1
INDEX=3 <==> run 1 of lr1 = 0.001 lr 0.01
```
and the indices would wrap around for additional runs, that is:
```
INDEX=4 <==> run 2 of lr1 = 0.01 lr2 = 0.1
INDEX=5 <==> run 2 of lr1 = 0.01 lr2 = 0.01
INDEX=6 <==> run 2 of lr1 = 0.001 lr2 = 0.1
INDEX=7 <==> run 2 of lr1 = 0.001 lr 0.01
```
etc... That is, INDEX=i mod (#settings) refers to the runs using settings combination i

An example command line: 
`python3 main.py --env_json jsonfiles/environment/Pendulum-v0.json --agent_json jsonfiles/agent/reverse_kl.json --index 0`

OR to train multiple runs using the same parameter settings:
`for i in $(seq 0 36 $(echo "36 * 10" | bc)); do python3 main.py --env_json jsonfiles/environment/Pendulum-v0.json --agent_json jsonfiles/agent/reverse_kl.json --index $i; done
`

**Run:** `python3 main.py --env_json jsonfiles/environment/$ENV.json --agent_json jsonfiles/agent/$AGENT.json --index $INDEX`


(`--render` and `--monitor` is optional, to visualize/monitor the agents' training, only available for openai gym or mujoco environments. `--write_plot` is also available to plot the learned action-values and policy on Bimodal1DEnv domain.)


* ENV.json is used to specify evaluation settings:
  * TotalMilSteps: Total training steps to be run (in million)
  * EpisodeSteps: Steps in an episode (Use -1 to use the default setting)
  * EvalIntervalMilSteps: Evaluation Interval steps during training (in million)
  * EvalEpisodes: Number of episodes to evaluate in a single evaluation
  
* AGENT.json is used to specify agent hyperparameter settings: 
  * norm: type of normalization to use
  * exploration_policy: "ou_noise", "none": Use "none" if the algorithm has its own exploration mechanism
  * actor/critic l1_dim, l2_dim: layer dimensions
  * learning rate
  * other algorithm specific settings
  
