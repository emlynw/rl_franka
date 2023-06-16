import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames=3):
        self._env = env
        self._num_frames = num_frames
        self._pixel_shape = env.observation_space['pixels'].shape
        self._state_shape = env.observation_space['state'].shape
        self._pixel_frames = deque([], maxlen=num_frames)
        self._state_frames = deque([], maxlen=num_frames)
        self.observation_space = Dict({"state": Box(low=-np.inf, high=np.inf, shape=(num_frames, *self._state_shape), dtype=np.float32),
                                       "pixels": Box(low=0, high=255, shape=(num_frames, *self._pixel_shape), dtype=np.uint8)})
        self.action_space = env.action_space


    def step(self, action):

        obs, reward, terminated, truncated, info = self._env.step(action)
        state = obs['state']
        pixels = obs['pixels']
        self._state_frames.append(state)
        self._pixel_frames.append(pixels)
        stacked_states = np.array(list(self._state_frames))
        stacked_pixels = np.array(list(self._pixel_frames))
        obs['state'] = stacked_states
        obs['pixels'] = stacked_pixels

        return obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self._env.reset()
        state = obs['state']
        pixels = obs['pixels']
        for _ in range(self._num_frames):
            self._state_frames.append(state)
            self._pixel_frames.append(pixels)
        stacked_states = np.array(list(self._state_frames))
        stacked_pixels = np.array(list(self._pixel_frames))
        obs['state'] = stacked_states
        obs['pixels'] = stacked_pixels
        
        return obs, info
