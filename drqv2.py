import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.optim import AdamW

import utils

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, pixel_shape):
        super().__init__()

        assert len(pixel_shape) == 3
        self.repr_dim = 76832

        self.convnet = nn.Sequential(nn.Conv2d(pixel_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, state_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim + state_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, embs, states, std):
        h = self.trunk(embs)
        h = torch.cat([h, states], dim=-1)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    
class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, state_dim, hidden_dim, dropout=0.01):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + state_dim + action_shape[0], hidden_dim),
            nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + state_dim + action_shape[0], hidden_dim),
            nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=dropout), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, embs, states, action):
        h = self.trunk(embs)
        h_action = torch.cat([h, states, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class drqv2Agent(nn.Module):
  """Soft-Actor-Critic."""

  def __init__(self, device, pixel_dim, state_dim, action_shape):
    super().__init__()
    self.pixel_dim = pixel_dim
    self.state_dim = state_dim
    self.action_shape = action_shape
    self.device = device
    self.lr = 1e-4
    self.feature_dim = 50
    self.hidden_dim = 1024
    self.critic_target_tau = 0.01
    self.num_expl_steps = 3000
    self.num_seed_steps = 4000
    self.update_every_steps = 1
    self.stddev_schedule = 1.0
    self.stddev_clip = 0.3
    self.use_tb = True

    self.discount = 0.99
    self.nstep = 3
    self.capacity = 60_000
    self.num_train_steps = 200_000
    self.num_eval_episodes = 5
    self.eval_frequency = 10_000
    self.checkpoint_frequency = 20_000
    self.log_frequency = 1_000
    self.batch_size = 256
    self.dropout = 0.01
    self.utd = 2
      
    # models
    self.encoder = Encoder(self.pixel_dim).to(device)
    self.actor = Actor(self.encoder.repr_dim, self.action_shape, self.feature_dim,
                        self.state_dim, self.hidden_dim).to(device)

    self.critic = Critic(self.encoder.repr_dim, self.action_shape, self.feature_dim,
                          self.state_dim, self.hidden_dim).to(device)
    self.critic_target = Critic(self.encoder.repr_dim, self.action_shape, self.feature_dim, 
                                self.state_dim, self.hidden_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    # Optimizers.
    self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    # Augmentations
    self.aug = RandomShiftsAug(pad=4)

    self.train()
    self.critic_target.train()

  def train(self, training = True):
    self.training = training
    self.encoder.train(training)
    self.actor.train(training)
    self.critic.train(training)

  def act(self, pixels, states, step, eval_mode):
      pixels = torch.as_tensor(pixels, device=self.device)
      states = torch.as_tensor(states, device=self.device)
      states = states.unsqueeze(0)
      embs = self.encoder(pixels.unsqueeze(0))
      stddev = utils.schedule(self.stddev_schedule, step)
      dist = self.actor(embs, states, stddev)
      if eval_mode:
          action = dist.mean
      else:
          action = dist.sample(clip=None)
          if step < self.num_expl_steps:
              action.uniform_(-1.0, 1.0)
      return action.cpu().numpy()[0]

  def update_critic(self, embs, states, action, reward, discount, next_embs, next_states, step):
    metrics = dict()

    reward = reward.unsqueeze(-1)
    discount = discount.unsqueeze(-1)


    with torch.no_grad():
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(next_embs, next_states, stddev)
        next_action = dist.sample(clip=self.stddev_clip)
        target_Q1, target_Q2 = self.critic_target(next_embs, next_states, next_action)
        target_V = torch.min(target_Q1, target_Q2)
        target_Q = reward + (discount * target_V)

    Q1, Q2 = self.critic(embs, states, action)
    critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)


    if self.use_tb:
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

    # optimize encoder and critic
    self.encoder_opt.zero_grad(set_to_none=True)
    self.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    self.critic_opt.step()
    self.encoder_opt.step()

    return metrics

  def update_actor(self, embs, states, step):
    metrics = dict()

    stddev = utils.schedule(self.stddev_schedule, step)
    dist = self.actor(embs, states, stddev)
    action = dist.sample(clip=self.stddev_clip)
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    Q1, Q2 = self.critic(embs, states, action)
    Q = torch.cat((Q1, Q2))

    actor_loss = -Q.mean()

    # optimize actor
    self.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    self.actor_opt.step()

    if self.use_tb:
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

    return metrics

  def update(self, replay_iter, step):
    metrics = dict()

    if step % self.update_every_steps != 0:
        return metrics
    
    for i in range(self.utd):

        batch = next(replay_iter)
        pixels, states, action, reward, discount, next_pixels, next_states = utils.to_torch(
            batch, self.device)

        # augment
        pixels = self.aug(pixels.float())
        next_pixels = self.aug(next_pixels.float())
        # encode
        embs = self.encoder(pixels)
        with torch.no_grad():
            next_embs = self.encoder(next_pixels)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(embs, states, action, reward, discount, next_embs, next_states, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                self.critic_target_tau)
        
    # update actor
    batch = next(replay_iter)
    pixels, states, action, reward, discount, next_pixels, next_states = utils.to_torch(batch, self.device)
    metrics.update(self.update_actor(embs.detach(), states.detach(), step))

    return metrics

  # def optim_dict(self):
  #   return {
  #       "encoder_optimizer": self.encoder_optimizer,
  #       "actor_optimizer": self.actor_optimizer,
  #       "critic_optimizer": self.critic_optimizer,
  #   }