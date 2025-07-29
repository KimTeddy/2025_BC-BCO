# pip install stable-baselines[mpi]
# https://stable-baselines.readthedocs.io/en/master/guide/install.html
#conda install tensorflow

#wsl
#conda create -n baselines python=3.10
#conda activate baselines
#pip install "numpy<2" tensorflow
#sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
#pip install stable-baselines[mpi]
#pip uninstall numpy
#pip install "numpy<2"

# from stable_baselines import DQN
# from stable_baselines.gail import generate_expert_traj

# model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
#       # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
#       # data will be saved in a numpy archive named `expert_cartpole.npz`
# generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)