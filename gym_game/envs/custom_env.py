import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D
from gym_game.envs.observer import Observer

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self, observer):
        self.pygame = PyGame2D(None)
        self.action_space = spaces.MultiBinary(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=np.int)
        self.observer = observer

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D(self.observer)
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
