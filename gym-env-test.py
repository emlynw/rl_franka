import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, TimeLimit
import cv2

import gym_INB0104
import numpy as np

def main():
    env = gym.make("gym_INB0104/INB0104-v0", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)
    env = PixelObservationWrapper(env, pixels_only=False)
    

    # reset the environment
    obs, info = env.reset()
    cv2.imshow("obs", cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2BGR))
    cv2.waitKey(10)

    while True:
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # action = np.zeros(len(env.action_space.sample()))
            action = np.array([0, 0, -0.5, 1])
            obs, reward, terminated, truncated, info = env.step(action)
            cv2.imshow("obs", cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)
        print(F"done: {terminated}, truncated: {truncated}")
        obs, info = env.reset()
        cv2.imshow("obs", cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
