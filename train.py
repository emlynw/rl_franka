import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import collections
from tqdm.auto import tqdm
import psutil
import cv2
from drqv2 import drqv2Agent
from replay_buffer import ReplayBufferStorage, make_replay_loader
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
from wrappers import ActionRepeat, VideoRecorder, CustomObservation, FrameStackWrapper
import gym_INB0104
from gymnasium.spaces import Box, Dict
import utils

class Workspace:
  def __init__(self):
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.num_gpus = torch.cuda.device_count()
    cwd = os.getcwd()
    workdir = Path.cwd()
    self.work_dir = workdir
    tb_path = os.path.join(cwd, 'tb')
    cp_path = os.path.join(cwd, 'checkpoints')
    os.makedirs(tb_path, exist_ok=True)
    os.makedirs(cp_path, exist_ok=True)
    self.writer = SummaryWriter(log_dir=tb_path)
    self.frame_stack = 3
    self.action_repeat = 2
    self.ep_len = 500
  
    self.setup()

  def setup(self):
    self.env = self.create_environment(name="gym_INB0104/INB0104-v0",frame_stack=self.frame_stack, action_repeat=self.action_repeat)
    self.eval_env = self.create_environment(name="gym_INB0104/INB0104-v0", frame_stack=self.frame_stack, action_repeat=self.action_repeat, record=True)
    self.policy = drqv2Agent(self.device, self.env.observation_space.shape, self.env.action_space.shape)
    # create replay buffer
    self.work_dir = Path.cwd()

    self.replay_storage = ReplayBufferStorage(self.work_dir / 'buffer')

    self.replay_loader = make_replay_loader(
        self.work_dir / 'buffer', self.policy.capacity,
        self.policy.batch_size, 1,
        False, 3, self.policy.discount)
    self._replay_iter = None

  @property
  def replay_iter(self):
      if self._replay_iter is None:
          self._replay_iter = iter(self.replay_loader)
      return self._replay_iter
      
  def create_environment(self, name, frame_stack=3, action_repeat=2, record=False, video_dir="./eval_vids"):
    
    env = gym.make(name, render_mode='rgb_array')
    if action_repeat > 1:
      env = ActionRepeat(env, action_repeat)
    if record:
      env = VideoRecorder(env, save_dir=video_dir, crop_resolution=480, resize_resolution=224)
    env = PixelObservationWrapper(env, pixels_only=True)
    env = CustomObservation(env, crop_resolution=480, resize_resolution=112)
    env = FrameStackWrapper(env, frame_stack)

    return env

  def eval(self, i):
    stats = collections.defaultdict(list)
    for j in range(self.policy.num_eval_episodes):
      obs, info = self.eval_env.reset()
      terminated = False
      truncated = False
      total_reward = 0
      while not (terminated or truncated):
        with torch.no_grad(), utils.eval_mode(self.policy):
          action = self.policy.act(obs, i, eval_mode=True)
        obs, reward, terminated, truncated, info = self.eval_env.step(action)
        total_reward += reward
      end_reward = reward
      stats["end_reward"].append(end_reward)
      stats["episode_reward"].append(total_reward)
    for k, v in stats.items():
      stats[k] = np.mean(v)
    return stats
  
  def train(self):
    try:
      obs, info = self.env.reset()
      action = self.env.action_space.sample()
      reward = np.float32(-1.0)
      terminated = False
      truncated = False
      episode_reward = 0
      time_step = {"observation": obs, "action": action, "reward": reward, "discount": 1.0, "truncated": truncated}
      self.replay_storage.add(time_step)
      for i in tqdm(range(self.policy.num_train_steps)):
        if terminated or truncated:
          self.writer.add_scalar("episode end reward", reward, i)
          self.writer.add_scalar("episode return", episode_reward, i)
          # Reset env
          time_step = self.env.reset()
          terminated = False
          truncated = False
          time_step = {"observation": obs, "action": action, "reward": reward, "discount": 1.0, "truncated": truncated}
          self.replay_storage.add(time_step)
          episode_reward = 0

        # Evaluate
        if i  % self.policy.eval_frequency == 0:
          eval_stats = self.eval(i)
          for k, v in eval_stats.items():
            self.writer.add_scalar(f"eval {k}", v, i)

        # Sample action
        with torch.no_grad(), utils.eval_mode(self.policy):
          action = self.policy.act(obs, i, eval_mode=False)
          
        # Update agent
        if i >= self.policy.num_seed_steps:
          train_info = self.policy.update(self.replay_iter, i)
          if i % self.policy.log_frequency == 0:
            if train_info is not None:
              for k, v in train_info.items():
                self.writer.add_scalar(k, v, i)
            ram_usage = psutil.virtual_memory().percent
            self.writer.add_scalar("ram usage", ram_usage, i)

        # Take env step
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.astype(np.float32)
        if terminated:
          discount = 0.0
        else:
          discount = 1.0
        episode_reward += reward
        time_step = {"observation": obs, "action": action, "reward": reward, "discount": discount, "truncated": truncated}
        self.replay_storage.add(time_step)


    except KeyboardInterrupt:
      print("Caught keyboard interrupt. Saving before quitting.")

    finally:
      print(f"done?")  # pylint: disable=undefined-loop-variable


def main():
  workspace = Workspace()
  workspace.train()

if __name__ == "__main__":
  main()
