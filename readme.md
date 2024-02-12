### Franka RL
Mujoco simulation of Franka B in INB0104 for use in sim to real RL training. The package contains two environments, a cartesian velocity control and joint velocity control for performing a pushing task.

### Installation:

Requires: mujoco, gymnasium

clone the repository
cd rl_franka/gym-INB0104
pip3 install -e .

### Running the current simulation:
python3 gym-env-test.py

### editing and adding to the simulation:
- The Gymnasium RL training script is found at /src/gym-env-test.py
- The gymnasium simulation environment is found in /gym-INB0104/envs/INB0104
- The INB0104 .xml head script is found in /environments/INB0104/Robot_C.xml



