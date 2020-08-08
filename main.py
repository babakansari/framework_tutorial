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


def in_confidence_interval(e, percent):
    return e > percent and e < (1-percent)

def action_from_sensors(state):
    direction = np.argmax(state)
    if direction == 2:
        return 0
    elif direction < 2 :
        return 2
    else:
        return 1

# Policy take a state and returns an action
def policy(state, q_table):
    global epsilon
    rand = random.uniform(0, 1)
    if( in_confidence_interval(rand, 0.05) and np.std(state)>2.5 ):
        print("State variance %f" % (np.std(state)))
        return action_from_sensors(state) # Confident move
    else:
        if rand < epsilon:
            return env.action_space.sample() # Exploration
        else:
            return np.argmax(q_table[state]) # Exploitation

def train(q_table):
    global epsilon, learning_rate, render_detail_move
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # Policy is to do random action to learn at begining
            action = policy(state, q_table)

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
                print("Episode %d on %i steps with reward = %f and %f epsilon" % (episode, t, total_reward, epsilon))
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
    render_detail_move = True
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
    train( q_table )

    # Save Q-Table
    save_training( q_table )
    
    # Apply learnt knowldge
    apply_learning( q_table )
