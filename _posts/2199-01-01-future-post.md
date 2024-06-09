---
title: 'Playing Atari Games with Reinforcement Learning'
date: 2024-06-15
permalink: /posts/2012/08/blog-post-4/
tags:
  - AI
  - Reinforcement Learning
  - Computer Science
---

In this blog post, we will explore how to train agents to play classic Atari games using Reinforcement Learning (RL). We will focus on the Double Deep Q-Network (DDQN) algorithm and showcase its results in games like Mini-Breakout, Breakout, and Pong.

## Introduction to Reinforcement Learning

Reinforcement Learning is a branch of machine learning where an agent learns to make decisions by interacting with an environment. The agent's goal is to maximize cumulative reward over time. This type of learning is based on the concept of trial and error, where the agent receives feedback in the form of rewards or punishments based on the actions it takes.

### Components of RL

1. **Agent**: The entity that makes decisions.
2. **Environment**: The world with which the agent interacts.
3. **Actions**: The decisions that the agent can make.
4. **Rewards**: Feedback signals from the environment indicating the success of the agent's actions.
5. **Policy**: The agent's strategy for making decisions based on the current state of the environment.

| ![Reinforcement Learning Process](../../../../images/rl_diagram.png) | 
|:--:| 
| *Figure 1: Reinforcement Learning Process.* |

## Double Deep Q-Network (DDQN)

The Double Deep Q-Network (DDQN) algorithm is an improvement over the Deep Q-Network (DQN) algorithm, which itself is a RL technique that uses deep neural networks to approximate the Q-value function. The DQN algorithm has some issues, such as overestimating Q-values, which can lead to suboptimal policies. DDQN addresses this problem by using two neural networks instead of one. To learn more about Reinforcement Learning concepts and algorithms, be sure to check out my other [blog posts](https://maxgalindo150.github.io/blog/year-archive/).

### How DDQN Works

1. **Main Q Network and Target Q Network**: DDQN uses two Q networks. The main network is updated at every time step, while the target network is updated periodically.
2. **Q-Value Update**: Instead of using the main Q network to both select and evaluate an action, DDQN uses the main network to select the action and the target network to evaluate it.
3. **Loss Function**: The loss function is calculated as follows:

   $$
   L(\theta) = \mathbb{E}\left[\left(r + \gamma Q(s', \text{argmax}_{a'} Q(s', a'; \theta); \theta^{-}) - Q(s, a; \theta)\right)^2\right]
   $$

   Where:
   - $$ r $$ is the reward received after taking action $$ a $$.
   - $$ \gamma $$ is the discount factor.
   - $$ s' $$ is the next state.
   - $$ \theta $$ are the parameters of the main network.
   - $$ \theta^{-} $$ are the parameters of the target network.

## Network Architecture

For the network architecture, we based our design on the article "Dueling Network Architectures for Deep Reinforcement Learning" by Wang et al. This architecture separates the representation of state values and advantages, which helps the agent to learn more efficiently.

### Dueling Network Architecture

The Dueling Network Architecture consists of two streams that represent the value and advantage functions separately:

1. **Value Stream**: This stream estimates the value of being in a given state, regardless of the action taken. It outputs a scalar value $$V(s)$$.

2. **Advantage Stream**: This stream estimates the advantage of each action in a given state, representing the relative importance of each action. It outputs a vector of advantages $$ A(s, a) $$ for each possible action.

The final Q-values are then computed by combining these two streams as follows:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left(A(s, a; \theta, \alpha) - \frac{1}{\|\mathcal{A}\|} \sum_{a'} A(s, a'; \theta, \alpha)\right)
$$

Where:
- $$ \theta $$ represents the parameters of the shared convolutional layers.
- $$ \alpha $$ and $$ \beta $$ are the parameters of the advantage and value streams, respectively.
- $$ \|\mathcal{A}\| $$ is the number of possible actions.

This architecture helps to reduce the problem of overestimating Q-values and leads to more stable and efficient learning.

| ![Reinforcement Learning Process](../../../../images/net.png) | 
|:--:| 
| *Figure 2: Dueling Network Architecture* |

## Experiments with Atari Games

To demonstrate the effectiveness of the DDQN algorithm, we trained it on three classic Atari games: Mini-Breakout, Breakout, and Pong. Below are the training graphs and a video showing the agents playing these games once trained.

### Mini-Breakout

| ![Training Mini-Breakout](../../../../images/train_min.png) | 
|:--:| 
| *Figure 3: Training for Mini-Breakout* |

[Here you can watch a video of the trained agents playing Mini-Breakout](https://youtu.be/LusMR9KKoaM?si=LOilcSJgcOWA2H76)

### Breakout

| ![Training Breakout](../../../../images/train_brek.png) | 
|:--:| 
| *Figure 4: Training for Breakout* |

[Here you can watch a video of the trained agents playing Atari-Breakout](https://youtu.be/DnYy1ND9zN8?si=x9xauw56d0xSdjAU)

### Pong

| ![Training Mini-Breakout](../../../../images/train_pong.png) | 
|:--:| 
| *Figure 5: Training for Pong* |

[Here you can watch a video of the trained agents playing Atari-Pong](https://youtu.be/VN4_u7LbtnY?si=qtruJQBsCackQHst)

## Conclusions

The DDQN algorithm proves to be a powerful tool for training agents in Atari games, overcoming some of the limitations of the original DQN. Through this process, we have seen how an agent can effectively learn to play by interacting with the environment and optimizing a decision-making policy.

Thank you for reading! I hope this post has provided you with a clear understanding of Reinforcement Learning and the DDQN algorithm.

To access the source code, visit my [GitHub repository](https://github.com/MaxGalindo150/Reinforcement_Learning_Projects/tree/master/Deep_Reinforcement_Learning/DDQN).


## References

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
- Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.
