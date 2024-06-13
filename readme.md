### Franka RL
Mujoco simulation of Franka B in INB0104 for use in sim to real RL training. The package contains two environments, a cartesian velocity control and joint velocity control for performing a pushing task.

### Installation:

Requires: mujoco, gymnasium

clone the repository
cd rl_franka/gym_INB0104
pip3 install -e .

### Running the current simulation:
python /rl_franka/gym_INB0104/test/cartesian_velocity_test.py
python /rl_franka/gym_INB0104/test/joint_velocity_test.py

### editing and adding to the simulation:
- The gymnasium simulation environments are found in /rl_franka/gym_INB0104/gym_INB0104/envs
- The mujoco xml files are in /rl_franka/gym_INB0104/gym_INB0104/environments/INB0104


