<!-- Adding ceil brackets for latex display -->
$\def\lc{\left\lceil}   
\def\rc{\right\rceil}$

# Learning Liar's dice with deep Q-learning



A reinforcement learning application to Liar's Dice.

## To do list
- [ ] Add mlflow ?
- [ ] Streamlit app
- [ ] Convert bet into tuples. 

## Overview
This project explores the application of deep reinforcement learning to Liar’s Dice, an imperfect-information, turn-based bluffing game.  
This document presents the game mechanics, key reinforcement learning concepts, and their application to this setting.  
The [demo](demo.ipynb) notebook shows how to use the code to simulate games and train an agent using deep reinforcement learning.

**Results highlight:** The trained DQN agent achieves ~75% first place rate vs. two simple rule-based opponents after 5000 training games. (See [Result analysis](#result_analysis) for plots and metrics.)



## Key Features
- Custom Liar’s Dice game environment in Python.
- Deterministic and random baseline opponents.
- Visual tools for data analysis.
- Reinforcement learning modeling (state format, episode definition, reward definition).
- DDQN (double DQN) algorithm implementation in PyTorch.
- Context specific training and testing metrics.
- Detailed analysis in an interactive notebook.

## Table of contents
1. [Liar's dice game presentation](#game_presentation)  

2. [Reinforcement learning theory](#rl_theory)  
    2.1 [General idea](#general_idea)  
    2.2 [Q learning](#q_learning)  
    2.3 [DQN algorithm](#)  
    2.4 [DDQN update](#)

3. [Training an agant with reinforcement learning](#rl_application)  
    3.1 [Environment setup](#env_setup)  
    3.2 [RL framework application](#rl_framework_application)  
    3.3 [Training loop](#)  

4.[Result analysis](#result_analysis)


5. [Possible improvements](#possible_improvements)

## 1. Liar's dice game explanation <a name="game_presentation"></a>

Liar’s Dice is a turn-based bluffing game with imperfect information.  
Each player starts with 5 dice. At the beginning of each round, all players roll their dice privately, then take turns making a **bet** about the number of dice showing a certain value among all dice in play.

On their turn, a player must either **outbid** the previous **bet** or **challenge** it.


### Bet format
A bet is written as `[q, v]` and means: *there are at least `q` dice showing face `v` across all dice at play*.  
**Ones are wild**: when `v ≠ 1`, dice showing `1` also count toward as the face of current bet.

### Outbidding rules
Given the previous bet `[q, v]`, a legal outbid must satisfy **one** of the following conditions:

- Increase the face value while keeping the same quantity: `[q, v′]` with `v′ > v`
- Increase the quantity with any face: `[q′, v′]` with `q′ > q`

Additional constraints apply when switching between wilds and non-wilds:
- From wilds to non-wilds (`[q, 1] → [q′, v′ ≠ 1]`):  
  the new quantity must satisfy  
  `q′ ≥ 2q + 1`
- From non-wilds to wilds (`[q, v ≠ 1] → [q′, 1]`):  
  the quantity may decrease, but must satisfy  
  `q′ ≥ ceil(q / 2)`
  *(this is the only way to reduce the quantity in a bet)*

### Challenging previous player
Instead of outbidding, a player may challenge the previous bet by calling either **"liar"** *or **"exact"**. In this case, dice are revealed and previous bet validity is checked.

- **"Liar"**: 
  If the true count of dice showing `v` (and/or `1`) is **≥ q**, the challenger loses one die; otherwise the previous player loses one die.
- **"Exact"**: if the challenger calls *exact* and the true count equals `q`, the challenger gains one die back; otherwise the challenger loses one die.

### Note
Many Liar’s Dice variants exist; only the rules described above are implemented in this project.

## 2. Reinforcement learning theory <a name="rl_theory"></a>

### 2.1 Core idea <a name="core_idea"></a>
Reinforcement learning (RL) is a branch of machine learning. It is a framework for training agents to make decisions in an environment by interacting with it.
An RL problem is commonly modeled as a Markov Decision Process (MDP) defined by the tuple  
$(\mathcal{S}, \mathcal{A}, \mathbb{P}, r, \gamma)$, where:

- $\mathcal{S}$ is the state space and $\mathcal{A}$ the action space,
- $\mathbb{P}(s' \mid s, a)$ is the probability of transitioning from state $s$ to $s'$ after taking action $a$,
- $r(s,a,s')$ is the reward for taking action $a$ in state $s$ and transitioning to $s'$,
- $\gamma \in (0,1)$ is the discount factor.

At time $t$, the agent observes a state $S_t \in \mathcal{S}$, selects an action $A_t \in \mathcal{A}$ according to a policy  
$\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$, receives a reward $R_{t+1}$, and transitions to the next state $S_{t+1}$.

The **action-value function** (Q-function) under a policy $\pi$ is defined as the expected discounted sum of future rewards obtained by taking action $a$ in state $s$:
$$
Q_{\pi}(s,a) =
\mathbb{E}_{\pi}\Big[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}
\,\Big|\, S_t = s, A_t = a\Big].
$$

The goal of reinforcement learning is to find an optimal policy $\pi^*$ that maximizes the Q-function:
$$
Q_{\pi^*}(s,a) = \max_{\pi} Q_{\pi}(s,a).
$$

The optimal Q-function satisfies the **Bellman optimality equation**:
$$
Q_{\pi^*}(s,a) =
\mathbb{E}_{s' \sim \mathbb{P}(\cdot \mid s,a)}
\Big[r(s,a,s') + \gamma \max_{a'} Q_{\pi^*}(s',a')\Big].
$$

### 2.2 Q-learning <a name="q_learning"></a>
Q-learning is an algorithm made for finding the optimal policy $\pi^*$ by iteratively solving Bellman's optimality equation.

The main ideas are:
- listing every (state,action) pair in a table, refered to as the Q-table, and randomly initialize every Q-value Q(s,a) in it
- fixing $\pi$ so that the agent will always take the action with the highest Q-value: $A_t = argmax_{a}Q(s,a)$
 
## 3. Training an agant with reinforcement learning <a name="rl_application"></a>

### 3.1 Environment setup <a name="env_setup"></a>  

#### <u> Deterministic policies to beat :</u>
In order to represent three types of player, for the rl agent to play against, we defined **three distinct policies**:

- A "*survivalist*" policy that maximizes survival: each turn, this agent selects the action (calling "liar", "exact" or outbidding) with the highest probability of being true. In case of equal probabilities, the most aggressive action (i.e. with highest quantity and/or value ) is chosen.
- An "*aggressive*" policy: each turn, selects the action with the lowest probability of being true among those whose probability exceeds a predefined threshold (50% by default). This policy favors high quantity/value bets while maintaining a minimum survival probability.
- A "*random*" policy: each turn, returns a random (from uniform distribution) action among legal actions.

Action probabilities are computed using a **binomial distribution conditioned on the player’s hand**.


#### <u> Game setup : </u>
For simplicity and keeping training duration under an hour, results presented in current document were obtained with the following setup:
- the rl agent learnt by playing against **two deterministic agents**: agent_max_probability and agent_min_probability;
- every agent started with **two dice**.

With such parameters, the whole model training process (warm-up + training loop) took on average 30 minutes.

<ins>Note :</ins>
It is possible to increase deterministic agent number or number of dice but doing so will significantly increase training time. Training the rl agent against two deterministic agent took more than 3 hours.

Bets are couples of quantity/value $[q, v]$

### 3.2 RL framework application <a name="rl_framework_application"></a>  


### 3.3 RL Training loop <a name="training_loop"></a>  


Here are the parameters used in [demo](demo.ipynb):


#### <u> Deep learning parameters :</u>
- batch_size = 256
- learning_rate = 3e-3
- criterion = nn.SmoothL1Loss()
- metric = nn.L1Loss()
- weight_decay_rate = 1e-4

#### <u> Reinforcement learning parameters :</u>
| Situation     | Reward      |
| :-------------: | :-------------: |
| Agent called liar and was right | 1 |
| Agent was called liar or was called liar and lost a dice | -2 |
| Agent outbid without being challenged | 0 |
| Agent was challenged and challenger lost a dice chalenge | 1 |
| Agent outbid, next player called exact and earned a dice back| -0.5 |
| Game ended without the agent challenging or being challenged | 0.5 |


## 4. Result analysis <a name="result_analysis"></a>

## 5. Possible improvements <a name="possible_improvements"></a>

- Taking game end as the only termina state

## Sources
- PyTorch's reinforcement learning tutorial : https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- DDQN algorithm explanation : https://apxml.com/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/double-dqn-ddqn

## Licence