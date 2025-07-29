import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)

# 모델 학습
model.learn(total_timesteps=10000)

# 데이터 수집
expert_obs = []
expert_actions = []

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    expert_obs.append(obs)
    expert_actions.append(action)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        obs, _ = env.reset()

# 저장
np.savez("expert_data.npz", obs=np.array(expert_obs), actions=np.array(expert_actions))
print("Saved expert_data.npz")
