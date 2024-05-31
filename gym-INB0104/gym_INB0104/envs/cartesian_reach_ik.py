# https://github.com/rail-berkeley/serl/blob/e2065d673131af6699aa899a78159859bd17c135/franka_sim/franka_sim/envs/panda_pick_gym_env.py
import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
import mujoco
from gym_INB0104.controllers import opspace_3 as opspace
from typing import Optional, Any, SupportsFloat
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }


class cartesian_reach_ik(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": [ 
            "human",
            "rgb_array", 
            "depth_array"
        ], 
    }
    
    def __init__(
        self,
        image_obs=True,
        control_dt=0.05,
        physics_dt=0.001,
        width=480,
        height=480,
        render_mode="rgb_array",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, 
            image_obs=image_obs,
            **kwargs
        )

        self.image_obs = image_obs
        self.render_mode = render_mode

        if self.image_obs:
            self.observation_space = Dict(
                {
                    "state": Dict(
                        {
                            "panda/tcp_pos": Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": Dict(
                        {
                            "front": Box(
                                low=0,
                                high=255,
                                shape=(height, width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": Box(
                                low=0,
                                high=255,
                                shape=(height, width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        else:
            self.observation_space = Dict(
            {
                "state": Dict(
                    {
                        "panda/tcp_pos": Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/tcp_vel": Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/gripper_pos": Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        # "panda/joint_pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        # "panda/joint_vel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        # "panda/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
                        # "panda/wrist_force": specs.Array(shape=(3,), dtype=np.float32),
                        "block_pos": Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
            }
        )


        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/cartesian_reach_ik.xml")
        self._n_substeps = int(control_dt / physics_dt)
        self.frame_skip = 1

        MujocoEnv.__init__(
            self, 
            env_dir, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG, 
            camera_id=0, 
            **kwargs,
        )

        self.camera_id = (0, 1)
        self.action_space = Box(
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)
        self.setup()

    def setup(self):
        self._PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
        self._PANDA_XYZ = np.asarray([0.3, 0, 0.5])
        self._CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
        self._SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])
        self._panda_dof_ids = np.asarray(
            [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id
        self._block_z = self.model.geom("block").size[2]
        self.action_scale: np.ndarray = np.asarray([0.1, 1])
        
        # Arm to home position
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        mujoco.mj_forward(self.model, self.data)
        
        # Reset mocap body to home position
        tcp_pos = self.data.sensor("pinch_pos").data
        self.data.mocap_pos[0] = tcp_pos

        mujoco.mj_step(self.model, self.data)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

        self.cam_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cam0")
        self.init_cam_pos = self.model.body_pos[self.cam_body_id].copy()
        self.init_cam_quat = self.model.body_quat[self.cam_body_id].copy()

        self.light_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "light0")
        self.init_light_pos = self.model.body_pos[self.light_body_id].copy()

        self.plywood_tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "plywood")
        self.table_tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "table")
        self.plywood_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "plywood")
        self.init_plywood_rgba = self.model.mat_rgba[self.plywood_mat_id].copy()

        self.brick_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "brick_wall")
        self.init_brick_rgba = self.model.mat_rgba[self.brick_mat_id].copy()

        self.left_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_curtain")
        self.init_left_curtain_rgba = self.model.geom_rgba[self.left_curtain_geom_id].copy()
        self.right_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_curtain")
        self.init_right_curtain_rgba = self.model.geom_rgba[self.right_curtain_geom_id].copy()
        self.back_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "back_curtain")
        self.init_back_curtain_rgba = self.model.geom_rgba[self.back_curtain_geom_id].copy()

        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.default_obj_pos = np.array([0.5, 0, 1.1])
        self.default_obs_quat = np.array([1, 0, 0, 0])
        self.object_center_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_center_site")

        

    def reset_model(self):
        # Reset arm to home position.
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        mujoco.mj_forward(self.model, self.data)

        # Reset mocap body to home position.
        tcp_pos = self.data.sensor("pinch_pos").data
        self.data.mocap_pos[0] = tcp_pos

        # Move robot
        # ee_noise_x = np.random.uniform(low=0.0, high=0.12)
        # ee_noise_y = np.random.uniform(low=-0.2, high=0.2)
        # ee_noise_z = np.random.uniform(low=-0.4, high=0.1)
        # ee_noise = np.array([ee_noise_x, ee_noise_y, ee_noise_z])
        self.data.mocap_pos[0] = self._PANDA_XYZ

        # Add noise to camera position and orientation
        cam_pos_noise = np.random.uniform(low=[-0.05,-0.05,-0.02], high=[0.05,0.05,0.02], size=3)
        cam_quat_noise = np.random.uniform(low=-0.01, high=0.01, size=4)
        self.model.body_pos[self.cam_body_id] = self.init_cam_pos + cam_pos_noise
        self.model.body_quat[self.cam_body_id] = self.init_cam_quat + cam_quat_noise
        # Add noise to light position
        light_pos_noise = np.random.uniform(low=[-0.8,-0.5,-0.2], high=[1.2,0.5,0.2], size=3)
        self.model.body_pos[self.light_body_id] = self.init_light_pos + light_pos_noise
        # Change light levels
        light_0_diffuse_noise = np.random.uniform(low=0.1, high=0.8, size=1)
        light_1_diffuse_noise = np.random.uniform(low=0.1, high=0.3, size=1)
        self.model.light_diffuse[0][:] = light_0_diffuse_noise
        self.model.light_diffuse[1][:] = light_1_diffuse_noise
        # Randomize table color
        channel = np.random.randint(0,3)
        table_color_noise = np.random.uniform(low=-0.05, high=0.2, size=1)
        self.model.mat_texid[self.plywood_mat_id] = np.random.choice([self.plywood_tex_id, self.table_tex_id])
        self.model.mat_rgba[self.plywood_mat_id] = self.init_plywood_rgba
        self.model.mat_rgba[self.plywood_mat_id][channel] = self.init_plywood_rgba[channel] + table_color_noise
        # Randomize brick color
        channel = np.random.randint(0,3)
        brick_color_noise = np.random.uniform(low=-0.1, high=0.1, size=1)
        self.model.mat_rgba[self.brick_mat_id] = self.init_brick_rgba
        self.model.mat_rgba[self.brick_mat_id][channel] = self.init_brick_rgba[channel] + brick_color_noise
        # Randomize curtain alpha
        # alpha = np.random.choice([0.0, 1.0, np.random.uniform(low=0.1, high=0.9)])
        alpha = 0.0
        self.model.geom_rgba[self.left_curtain_geom_id][3] = alpha
        self.model.geom_rgba[self.right_curtain_geom_id][3] = alpha
        self.model.geom_rgba[self.back_curtain_geom_id][3] = alpha

        # Move object
        self.object_x_noise = np.random.uniform(low=-0.15, high=0.1)
        self.object_y_noise = np.random.uniform(low=-0.1, high=0.1)
        self.object_theta_noise = np.random.uniform(low=-0.5, high=0.5)
        self.data.qpos[9] = self.default_obj_pos[0] + self.object_x_noise
        self.data.qpos[10] = self.default_obj_pos[1] + self.object_y_noise
        self.data.qpos[12] = self.default_obs_quat[0] + self.object_theta_noise
        
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        mujoco.mj_forward(self.model, self.data)
        for _ in range(5*self._n_substeps):
            tau = opspace(
                model=self.model,
                data=self.data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self.data.mocap_pos[0],
                ori=self.data.mocap_quat[0],
                joint=self._PANDA_HOME,
                gravity_comp=True,
            )
            self.data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self.model, self.data)
        
        self._z_init = self.data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2
        
        return self._get_obs()

    def step(self, action):
        # Action
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x, y, z, grasp = action
        pos = self.data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self.action_scale[0]
        npos = np.clip(pos + dpos, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        g = self.data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self.action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self.data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self.model,
                data=self.data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self.data.mocap_pos[0],
                ori=self.data.mocap_quat[0],
                joint=self._PANDA_HOME,
                gravity_comp=True,
            )
            self.data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._get_reward(action)

        return obs, reward, False, False, info 
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _get_obs(self):
        obs = {}
        obs["state"] = {}

        tcp_pos = self.data.sensor("pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self.data.sensor("pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self.data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = self.data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _get_reward(self, action):
        block_pos = self.data.sensor("block_pos").data
        tcp_pos = self.data.sensor("pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        reward = 0.3 * r_close + 0.7 * r_lift
        info = dict(reward_close=r_close, reward_lift=r_lift)
        return reward, info

