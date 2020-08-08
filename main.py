import sys
import numpy as np
import math
import random
import pygame
import gym
import gym_game
from gym_game.envs.observer import Observer
import pickle
import json
import os
import numpy

# Policy take a state and returns an action
def policy(state):
    global epsilon, q_table
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Exploration

    return np.argmax(q_table[state]) # Exploitation

def train():
    global epsilon, learning_rate, render_detail_move, q_table
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # Policy is to do random action to learn at begining
            action = policy(state)

            # Do action and get result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])

            # Temporal Difference Method:
            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # Draw games
            if(render_detail_move == True):
                env.render()

            # When episode is done, print reward

            if done or t >= MAX_TRY - 1:
                print("Episode %d on %i steps with reward = %f and %f learning rate" % (episode, t, total_reward, learning_rate))
                break

            if exit_program:
                return
        
        if(render_detail_move == False):
            env.render()

        epsilon = get_rate(epsilon)
            
def get_rate(i):
    global epsilon_decay
    return (i * epsilon_decay)

def load_training():
    if os.path.isfile('q_table.npy'):
        return numpy.load('q_table.npy')
    return np.zeros(num_box + (env.action_space.n,))

def save_training(q_table):
    numpy.save("q_table.npy", q_table)

def s_key_pressed(key):
    global render_detail_move, exit_program
    if( key ==  pygame.K_s):
        render_detail_move = not render_detail_move
    if( key ==  pygame.K_ESCAPE):
        exit_program = True

def apply_learning(q_table):
    global render_detail_move, exit_program
    exit_program = False
    state = env.reset()
    for t in range(MAX_EPISODES):
        action = np.argmax(q_table[state]) # Exploitation
        next_state, reward, done, _ = env.step(action)
        # Set up for the next iteration
        state = next_state
        if exit_program or done:
            break
        env.render()

#-- Program main 
if __name__ == "__main__":
    env = gym.make("Babak-v0", observer=Observer(s_key_pressed))
    render_detail_move = False
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6

    exit_program = False
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    # Load Q-Table
    q_table = load_training( )

    # Training the model
    train()

    # Save Q-Table
    save_training( q_table )
    
    # Apply learnt knowldge
    apply_learning( q_table )
