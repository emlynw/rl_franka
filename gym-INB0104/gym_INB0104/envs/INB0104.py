import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gym_INB0104.envs import mujoco_utils


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
    
    def __init__(self, render_mode=None, cartesian=True, use_distance=False, **kwargs):
        utils.EzPickle.__init__(self, cartesian, use_distance, **kwargs)
        self.use_distance = use_distance
        self.cartesian = cartesian
        if self.cartesian:
            if use_distance:
                observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
            else:
                observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        else:
            if use_distance:
                observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
            else:
                observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)

        env_dir = os.path.join(os.getcwd(), "environments/INB0104/Robot_C.xml")
        MujocoEnv.__init__(self, env_dir, 5, observation_space=observation_space, default_camera_config=DEFAULT_CAMERA_CONFIG, camera_id=0, **kwargs,)
        self.render_mode = render_mode
        # Leave these in in case useful later
        self.max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0])
        self.min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0])
        self.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 1.0])
        self.max_torque = np.array([87, 87, 87, 87, 12, 12, 12, 12, 1.0])

        self.default_robot_joint_qpos = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00, 0.655, 0.515, 0.94, 0, 0, 0, 1])
        self.default_obj_qpos = np.array([0.655, 0.515, 0.94, 0, 0, 0, 1])
        # self.default_obj_qpos = np.array([0.5, 0, 1.2, 0, 0, 0, 1])
        self.default_qpos = np.concatenate([self.default_robot_joint_qpos, self.default_obj_qpos])
        self._utils = mujoco_utils

    def step(self, a):
        target_pos = self.get_body_com("target_object")
        vec = self.get_body_com("ee_center_body")- target_pos
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        print(dir(self._model_names))

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
            dict(reward_dist=reward_dist, reward_ori=reward_ori, reward_ctrl=reward_ctrl),
            )

    def reset_model(self):
        self._step_count = 0
        # set up random initial state for the robot - but keep the fingers in place
        if self.cartesian:
            self.set_mocap_pose
        qpos = self.default_qpos
        if not self.cartesian:
            qpos[0] += self.np_random.uniform(low=-1, high=1)
            qpos[1] += self.np_random.uniform(low=-1, high=1)
            qpos[2] += self.np_random.uniform(low=-1, high=1)
            qpos[3] += self.np_random.uniform(low=-1, high=1)
            qpos[4] += self.np_random.uniform(low=-1, high=1)
            qpos[5] += self.np_random.uniform(low=-1, high=1)
            qpos[6] += self.np_random.uniform(low=-1, high=1)

        self.goal_x = self.np_random.uniform(low=-0.25, high=0.25)
        self.goal_y = self.np_random.uniform(low=-0.4, high=0.4)
        qpos[9] = self.default_obj_pos[0] + self.goal_x
        qpos[10] = self.default_obj_pos[1] + self.goal_y
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[9:11] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.cartesian:
            ee_pos = self.get_body_com("ee_center_body")
            gripper_width = self.get_fingers_width()
            if self.use_distance:
                return np.concatenate([ee_pos, gripper_width, self.get_body_com("ee_center_body") - self.get_body_com("target_object")])
            else:
                return np.concatenate([ee_pos, gripper_width])
        else:
            robot_joint_positions = self.data.qpos[0:8].flat.copy()
            robot_joint_velocities = self.data.qvel[0:8].flat.copy()
            if self.use_distance:
                return np.concatenate([robot_joint_positions, robot_joint_velocities, self.get_body_com("ee_center_body") - self.get_body_com("target_object")])
            else:
                return np.concatenate([robot_joint_positions, robot_joint_velocities])
            
    def _set_action(self, action) -> None:
        action = action.copy()
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])

        elif self.block_gripper:
            pos_ctrl = action
            fingers_half_width = 0

        # control the gripper
        self.data.ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)
            
    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
        
    
