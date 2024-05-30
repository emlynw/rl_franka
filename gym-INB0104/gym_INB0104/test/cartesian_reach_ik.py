import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    env = gym.make("gym_INB0104/cartesian_reach_ik", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)    
    camera_id = 1
    waitkey = 1

    # reset the environment
    obs, info = env.reset()
    pixels = env.render()[camera_id]
    cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    cv2.waitKey(waitkey)
    i=0

    while True:
        terminated = False
        truncated = False
        i=0
        while not terminated and not truncated:
            # action = np.zeros(len(env.action_space.sample()))
            if i < 25:
                action = np.array([0.1, -0.1, 0.1, 0.1])
            elif i >= 25 and i < 50:
                action = np.array([0.5, 0.1, -0.1, -0.1])
            elif i >= 50 and i < 75:
                action = np.array([0.5, 0.1, 0.1, 0.1])
            elif i >= 75 and i < 100:
                action = np.array([0.5, -0.1, -0.1, -0.1])
            obs, reward, terminated, truncated, info = env.step(action)
            pixels = env.render()[camera_id]
            cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            cv2.waitKey(waitkey)
            i+=1
        obs, info = env.reset()
        pixels = env.render()[camera_id]
        cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        cv2.waitKey(waitkey)


if __name__ == "__main__":
    main()
