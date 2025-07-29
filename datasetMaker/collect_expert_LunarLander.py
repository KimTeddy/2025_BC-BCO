import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=0, device="cpu")

# 모델 학습
model.learn(total_timesteps=10000)

# trajectory 저장
expert_obs_all = []
expert_actions_all = []

num_episodes = 500  # 수집할 에피소드 수
max_steps_per_episode = 200  # 최대 에피소드 길이

all_obs = []
all_actions = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    for _ in range(max_steps_per_episode):
        action, _ = model.predict(obs, deterministic=True)
        all_obs.append(obs)
        all_actions.append([action])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

# numpy array로 변환
expert_obs_all = np.array(all_obs, dtype=np.float32)        # shape: (총 step 수, 8)
expert_actions_all = np.array(all_actions, dtype=np.int32)  # shape: (총 step 수, 1)

# 저장
np.savez("expert_data_lunarlander.npz", obs=expert_obs_all, actions=expert_actions_all)
print("Saved expert_data_lunarlander.npz:", expert_obs_all.shape, expert_actions_all.shape)