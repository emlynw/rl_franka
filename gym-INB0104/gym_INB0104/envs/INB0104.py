import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
from gym_INB0104.envs import mujoco_utils
from typing import Optional, Any, SupportsFloat


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
        self._utils = mujoco_utils
        self.n_substeps = 20
        self.setup()

    def setup(self):
        self.default_obj_pos = np.array([0.5, 0, 1.2])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.04, 0.04])
        self.data.ctrl[0:7] = self.neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.init_qpos = self.neutral_joint_values
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._mujoco_step()


    def step(self, a):
        target_pos = self.get_body_com("target_object")
        vec = (self.get_body_com("left_finger")+self.get_body_com("right_finger"))/2 - target_pos
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        num_contacts = self.data.ncon
        reward_cont = -max(0,num_contacts-4)
        quat = self.data.xquat[10]
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])
        upright_orientation = np.array([0, 1, 0, 0])
        reward_ori = -np.linalg.norm(quat - upright_orientation)

        reward = reward_dist + 0.2*reward_cont

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
        noise = np.random.uniform(low=-1, high=1, size=(7,))
        qpos[0:7] += noise

        self.goal_x = self.np_random.uniform(low=-0.25, high=0.25)
        self.goal_y = self.np_random.uniform(low=-0.4, high=0.4)
        qpos[9] = self.default_obj_pos[0] + self.goal_x
        qpos[10] = self.default_obj_pos[1] + self.goal_y
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[9:11] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        robot_pos = self.data.qpos[0:8].flat.copy()
        robot_vel = self.data.qvel[0:8].flat.copy()
        if self.use_distance:
            return np.concatenate([robot_pos, robot_vel, self.get_body_com("ee_center_body") - self.get_body_com("target_object")])
        else:
            return np.concatenate([robot_pos, robot_vel])
        
    # Utils copied from https://github.com/zichunxx/panda_mujoco_gym/blob/master/panda_mujoco_gym/envs/panda_env.py
        
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == 1:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat
    
    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
