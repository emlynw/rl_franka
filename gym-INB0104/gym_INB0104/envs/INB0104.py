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
        self.action_scale = 0.05
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.04, 0.04])
        self.default_obj_pos = np.array([0.5, 0, 1.1])
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float64,
        )
        self.ep_steps = 0
        self.setup()

    def setup(self):
        self.set_joint_neutral()
        self.data.ctrl[0:7] = self.neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = np.array([0, 1, 0, 0])
        # self.grasp_site_pose = self.get_ee_orientation().copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        
        self._mujoco_step()
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def reset_model(self):
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        self.set_joint_neutral()
        ee_noise = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.set_mocap_pose(self.initial_mocap_position+ee_noise, self.grasp_site_pose)

        self.goal_x_noise = self.np_random.uniform(low=-0.25, high=0.25)
        self.goal_y_noise = self.np_random.uniform(low=-0.4, high=0.4)
        self.data.qpos[9] = self.default_obj_pos[0] + self.goal_x_noise
        self.data.qpos[10] = self.default_obj_pos[1] + self.goal_y_noise
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()

    def step(self, action):
        # Action
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._mujoco_step()
        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        # Reward
        target_pos = self.get_body_com("target_object").copy()
        vec = self.get_body_com("ee_center_body").copy() - target_pos
        r_dist = -np.linalg.norm(vec)
        r_ctrl = -np.square(action).sum()
        num_colls = self.data.ncon
        r_colls = -max(0,num_colls-4)
        quat = self.data.xquat[10]
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])
        upright_orientation = np.array([1, 0, 1, 0])
        r_ori = -np.linalg.norm(quat - upright_orientation)
        reward = r_dist + 0.2*r_colls
        info = dict(reward_dist=r_dist, reward_ori=r_ori)

        return obs, reward, False, False, info 
    
    def _set_action(self, action):
        action = action.copy()
        pos_ctrl = action[:3]
        gripper_ctrl = action[3]
        # Control gripper
        self.data.ctrl[-1] = gripper_ctrl
        # Change ee position
        pos_ctrl *= self.action_scale
        pos_ctrl += self.get_body_com("ee_center_body").copy()
        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

    def _get_obs(self):
        ee_pos = self.get_body_com("ee_center_body")
        gripper_width = self.get_fingers_width()
        if self.use_distance:
            target_pos = self.get_body_com("target_object")
            distance = np.linalg.norm(target_pos - ee_pos)
            return np.concatenate([ee_pos, gripper_width, [distance]])
        else:
            return np.concatenate([ee_pos, gripper_width])
        
    # Utils copied from https://github.com/zichunxx/panda_mujoco_gym/blob/master/panda_mujoco_gym/envs/panda_env.py
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == 1:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        self.data.qpos[0:9] = self.neutral_joint_values

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat
    
    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
