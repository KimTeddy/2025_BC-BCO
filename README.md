# 2025_BC-BCO
My first study of Behavior Cloning
- BC: Behavior Cloning
- BCO: Behavior Cloning from Observation

## Original Code
https://github.com/montaserFath/BCO<br>
➡️ Last commit was on Dec 15, 2020

## What did I do
- Getting Old Code to Work in 2025
  - Version Upgrades
- Convert Jupyter Notebook code to Python script

## Still Learning
- [BCO_Pendulum.py](BCO_Pendulum.py)

## Installing
```cmd
conda create -n bco python=3.10 -y
conda activate bco

conda install -y -c conda-forge numpy=1.23.5 gym ipython matplotlib
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8

pip install "gymnasium[box2d]"
```
[Classic Control](https://gymnasium.farama.org/environments/classic_control/)
- https://gymnasium.farama.org/environments/classic_control/pendulum/
- https://gymnasium.farama.org/environments/classic_control/cart_pole/

[Box2D](https://gymnasium.farama.org/environments/box2d/)
- https://gymnasium.farama.org/environments/box2d/lunar_lander/

[MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- https://gymnasium.farama.org/environments/mujoco/half_cheetah/