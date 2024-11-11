import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    max_future_q = np.max(Q[sprime])  
    current_q = Q[s, a] 
    new_q = current_q + alpha * (r + gamma * max_future_q - current_q)
    Q[s, a] = new_q
    
    return Q



def epsilon_greedy(Q, s, epsilone):
    if random.uniform(0, 1) < epsilone:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[s])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.8 # choose your own

    gamma = 0.7 # choose your own

    epsilon = 0.2 # choose your own

    n_epochs = 200 # choose your own
    max_itr_per_epoch = 100 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
