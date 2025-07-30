import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

n_envs = 9

# Parallel environments
vec_env = make_vec_env("Reacher-v5", n_envs=n_envs)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("ppo_test_Reacher")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_test_Reacher")

ep_cnt=0
success_cnt=0

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if dones.any():
        ep_cnt += dones.sum()
        success_cnt += rewards[dones].sum()
        
        if ep_cnt%100==0:
            print(f'success rate = {success_cnt/ep_cnt} ({success_cnt}/{ep_cnt})')