import sys
import numpy as np
import math
import random

import gym
import gym_game
from gym_game.envs.observer import Observer

def simulate():
    global epsilon, epsilon_decay, render_detail_move
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = np.argmax(q_table[state]) # Exploitation

            # Do action and get result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # Draw games
            if(render_detail_move == True):
                env.render()

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                break
        
        if(render_detail_move == False):
            env.render()

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay

def s_key_pressed():
    global render_detail_move
    render_detail_move = not render_detail_move

if __name__ == "__main__":
    env = gym.make("Babak-v0", observer=Observer(s_key_pressed))
    render_detail_move = False
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()
