import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0, device="cpu")

# 모델 학습
model.learn(total_timesteps=10000)

# trajectory 저장
expert_obs_all = []
expert_actions_all = []

num_episodes = 500  # 수집할 에피소드 수
max_steps_per_episode = 200  # CartPole의 최대 에피소드 길이

for ep in range(num_episodes):
    obs, _ = env.reset()
    episode_obs = []
    episode_actions = []

    for _ in range(max_steps_per_episode):
        action, _states = model.predict(obs, deterministic=True)
        episode_obs.append(obs)
        episode_actions.append([action])  # Discrete일 경우 1차원으로 묶음

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # if done:
        #     break

    expert_obs_all.append(episode_obs)
    expert_actions_all.append(episode_actions)

# 리스트 → numpy array로 변환 (padding 없이 variable length면 dtype=object 될 수 있음)
expert_obs_all = np.array(expert_obs_all, dtype=np.float32)
expert_actions_all = np.array(expert_actions_all, dtype=np.int32)

# 저장
np.savez("expert_data_cartpole.npz", obs=expert_obs_all, actions=expert_actions_all)
print("Saved expert_data_cartpole.npz:", expert_obs_all.shape, expert_actions_all.shape)



# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO

# env = gym.make("CartPole-v1")
# model = PPO("MlpPolicy", env, verbose=0)

# # 모델 학습
# model.learn(total_timesteps=10000)

# # 데이터 수집
# expert_obs = []
# expert_actions = []

# obs, _ = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     expert_obs.append(obs)
#     expert_actions.append(action)

#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     if done:
#         obs, _ = env.reset()

# # 저장
# np.savez("expert_data.npz", obs=np.array(expert_obs), actions=np.array(expert_actions))
# print("Saved expert_data.npz")