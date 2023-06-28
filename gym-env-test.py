import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import cv2

import gym_INB0104
import numpy as np
np.set_printoptions(suppress=True)

def main():
    env = gym.make("gym_INB0104/INB0104-v0", render_mode="human")
    # env = PixelObservationWrapper(env, pixels_only=False)
    # env = FrameStack(env, num_frames=3)
    
    # # print the observation space
    print("Observation space:", env.observation_space)
    # # print the action space
    print("Action space:", env.action_space)

    # reset the environment
    obs, info = env.reset()
    
    high = True
    # render the environment
    while True:
        for i in range(1000):
            if i % 100 == 0:
                high = not high
            if high:
                action = env.action_space.high
            else:
                action = env.action_space.low
            obs, reward, terminated, truncated, info = env.step(action)

        env.reset()


if __name__ == "__main__":
    main()
