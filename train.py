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
from sac import SAC
from replay_buffer_np import ReplayBuffer
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, RecordEpisodeStatistics
from wrappers import ActionRepeat, FrameStack, VideoRecorder, CustomObservation
import gym_INB0104
from gymnasium.spaces import Box, Dict

class dinov2_obs(gym.ObservationWrapper):
  # Embed image using Dinov2
  def __init__(self, env):
    super().__init__(env)
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.model =torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
    self.model.eval()
    self._embedding_shape = self.model.embed_dim
    self._state_shape = env.observation_space['state'].shape
    for param in self.model.parameters():
            param.requires_grad = False
    self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    self.observation_space = Dict({"state": Box(low=-np.inf, high=np.inf, shape=self._state_shape, dtype=np.float32),
                                       "embeddings": Box(low=-np.inf, high=np.inf, shape=(self.model.embed_dim,), dtype=np.float32)})

  def observation(self, obs):
    pixels = obs['pixels']
    # cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    pixels = self.transform(pixels)
    pixels = pixels.unsqueeze(0).to(self.device)
    features = self.model(pixels)
    obs['embeddings'] = features[0].cpu().numpy()
    return obs

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
    self.policy = SAC(self.device, self.env)
    # create replay buffer

    self.buffer = ReplayBuffer(
        embs_shape = (self.env.observation_space['embeddings'].shape[-1],),
        obs_frame_stack=self.frame_stack,
        action_shape=self.env.action_space.shape,
        batch_size=self.policy.batch_size,
        num_eps=self.policy.capacity//self.ep_len,
        ep_len=self.ep_len,
        device=self.policy.device,
                          )
      
  def create_environment(self, name, frame_stack=3, action_repeat=2, record=False, video_dir="./eval_vids"):
    
    env = gym.make(name, render_mode='rgb_array')
    if action_repeat > 1:
      env = ActionRepeat(env, action_repeat)
    if record:
      env = VideoRecorder(env, save_dir=video_dir, crop_resolution=480, resize_resolution=240)
    env = PixelObservationWrapper(env, pixels_only=False)
    env = CustomObservation(env, crop_resolution=480, resize_resolution=224)
    env = dinov2_obs(env)
    env = FrameStack(env, frame_stack)

    return env

  def evaluate(self, policy, env):
    """Evaluate the policy and dump rollout videos to disk."""
    policy.eval()
    stats = collections.defaultdict(list)
    for j in range(self.policy.num_eval_episodes):
      observation, info = env.reset()
      embs = observation['embeddings']
      terminated = False
      truncated = False
      total_reward = 0
      while not (terminated or truncated):
        action = policy.act(embs.flatten(), sample=False)
        observation, reward, terminated, truncated, info = env.step(action)
        embs = observation['embeddings']
        total_reward += reward
      end_reward = reward
      stats["end_reward"].append(end_reward)
      stats["episode_reward"].append(total_reward)
    for k, v in stats.items():
      stats[k] = np.mean(v)
    return stats
  
  def train(self):
    try:
      episode_reward = 0
      observation, _ = self.env.reset()
      embs = observation['embeddings']
      action = self.env.action_space.sample()
      reward = -1.0
      mask = 1.0
      terminated = False
      truncated = False
      self.buffer.insert(embs[-1], action, reward, mask)

      for i in tqdm(range(self.policy.num_train_steps)):
        if terminated or truncated:
          self.writer.add_scalar("episode end reward", reward, i)
          self.writer.add_scalar("episode return", episode_reward, i)
          # Reset env
          obs, _ = self.env.reset()
          embs = obs['embeddings']
          episode_reward = 0
          self.buffer.insert(embs[-1], action, reward, mask)

      # Evaluate
        if i % self.policy.eval_frequency == 0:
          eval_stats = self.evaluate(self.policy, self.eval_env)
          for k, v in eval_stats.items():
            self.writer.add_scalar(f"eval {k}", v, i)

        # Take action
        if i < self.policy.num_seed_steps:
          action = self.env.action_space.sample()
        else:
          action = self.policy.act(embs.flatten(), sample=True)

        # Update agent
        if i >= self.policy.num_seed_steps:
          self.policy.train()
          train_info = self.policy.update(self.buffer, i)

          if i % self.policy.log_frequency == 0:
            if train_info is not None:
              for k, v in train_info.items():
                self.writer.add_scalar(k, v, i)
            ram_usage = psutil.virtual_memory().percent
            self.writer.add_scalar("ram usage", ram_usage, i)

        # Take env step
        obs, reward, terminated, truncated, info = self.env.step(action)
        embs = obs['embeddings']
        episode_reward += reward

        if terminated:
          mask=1.0
        else: 
          mask=0.0
        
        self.buffer.insert(embs[-1], action, reward, mask)      


    except KeyboardInterrupt:
      print("Caught keyboard interrupt. Saving before quitting.")

    finally:
      print(f"done?")  # pylint: disable=undefined-loop-variable

def main():
  workspace = Workspace()
  workspace.train()

if __name__ == "__main__":
  main()
