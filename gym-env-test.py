import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import cv2

import gym_INB0104
import numpy as np

def main():
    env = gym.make("gym_INB0104/INB0104-v0", render_mode="rgb_array")
    env = PixelObservationWrapper(env, pixels_only=False)
    # env = FrameStack(env, num_frames=3)
    
    # # print the observation space
    print("Observation space:", env.observation_space)
    # # print the action space
    print("Action space:", env.action_space)

    # reset the environment
    obs, info = env.reset()
    cv2.imshow("obs", cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)
    
    high = True
    # render the environment
    while True:
        for i in range(50):
            if i % 100 == 0:
                high = not high
            if high:
                action = env.action_space.high
            else:
                action = env.action_space.low
            obs, reward, terminated, truncated, info = env.step(action)
            cv2.imshow("obs", cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)

        env.reset()


if __name__ == "__main__":
    main()
