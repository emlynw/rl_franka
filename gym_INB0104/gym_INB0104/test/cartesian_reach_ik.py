import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/cartesian_reach_ik", render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=80)    
    camera_id = 1
    waitkey = 50
    intial_pos = np.array([0.3, 0, 0.5])

    # reset the environment
    obs, info = env.reset()
    pixels = env.render()[camera_id]
    if render_mode == "rgb_array":
        pixels = cv2.resize(pixels, (224, 224))
        cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
        cv2.waitKey(waitkey)
    i=0

    while True:
        terminated = False
        truncated = False
        i=0
        while not terminated and not truncated:
            action = np.zeros(len(env.action_space.sample()))
            if i < 20:
                action = np.array([0.0, -0.5, 0.0, -1.0])
            elif i < 40:
                action = np.array([0.0, 0.0, -0.5, 1.0])
            elif i < 60:
                action = np.array([0.5, 0.0, 0.0, -1.0])
            elif i < 80:
                action = np.array([0.0, 0.5, 0.0, 1.0])
                

            obs, reward, terminated, truncated, info = env.step(action)
            pixels = env.render()[camera_id]
            if render_mode == "rgb_array":
                pixels = cv2.resize(pixels, (224, 224))
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            i+=1
        obs, info = env.reset()
        pixels = env.render()[camera_id]
        if render_mode == "rgb_array":
            pixels = cv2.resize(pixels, (224, 224))
            cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)


if __name__ == "__main__":
    main()
