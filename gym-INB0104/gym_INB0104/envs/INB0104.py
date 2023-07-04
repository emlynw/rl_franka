import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }


class INB0104Env(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": [ 
            "human",
            "rgb_array", 
            "depth_array"
        ], 
        "render_fps": 100
    }
    
    def __init__(self, render_mode=None, use_distance=False, **kwargs):
        utils.EzPickle.__init__(self, use_distance, **kwargs)
        self.use_distance = use_distance
        observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        cdir = os.getcwd()
        env_dir = os.path.join(cdir, "environments/INB0104/Robot_C.xml")
        MujocoEnv.__init__(self, env_dir, 5, observation_space=observation_space, default_camera_config=DEFAULT_CAMERA_CONFIG, camera_id=0, **kwargs,)
        self.render_mode = render_mode
        # Leave these in in case useful later
        self.max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0])
        self.min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0])
        self.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 1.0])
        self.max_torque = np.array([87, 87, 87, 87, 12, 12, 12, 12, 1.0])

    def step(self, a):
        target_pos = self.get_body_com("target_object")
        target_pos[2] += 0.1
        
        vec = self.get_body_com("left_finger") - target_pos
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        num_contacts = self.data.ncon
        contact_reward = -max(0,num_contacts-4)
        quat = self.data.xquat[10]
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])
        upright_orientation = np.array([0, 1, 0, 0])
        reward_ori = -np.linalg.norm(quat - upright_orientation)

        reward = reward_dist 

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob, 
            reward, 
            False, 
            False, 
            dict(reward_dist=reward_dist, reward_ori=reward_ori),
            )

    def reset_model(self):
        self._step_count = 0
        # set up random initial state for the robot - but keep the fingers in place
        qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04, 0.655, 0.515, 0.94, 0, 0, 0, 1])
        qpos[0] += self.np_random.uniform(low=-1, high=1)
        qpos[1] += self.np_random.uniform(low=-1, high=1)
        qpos[2] += self.np_random.uniform(low=-1, high=1)
        qpos[3] += self.np_random.uniform(low=-1, high=1)
        qpos[4] += self.np_random.uniform(low=-1, high=1)
        qpos[5] += self.np_random.uniform(low=-1, high=1)
        qpos[6] += self.np_random.uniform(low=-1, high=1)

        # create random x and y position for the target object, but make sure it is within a 1 meter circle -- this bit maybe useful later - not for now though
        while True:
            self.goal = self.np_random.uniform(low=-0.25, high=0.25, size=2)
            if np.linalg.norm(self.goal) < 1.0:
                break
        qpos[9:11] += self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[9:11] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        robot_pos = self.data.qpos[0:8].flat.copy()
        robot_vel = self.data.qvel[0:8].flat.copy()
        if self.use_distance:
            return np.concatenate([robot_pos, robot_vel, self.get_body_com("left_finger") - self.get_body_com("target_object")])
        else:
            return np.concatenate([robot_pos, robot_vel])
