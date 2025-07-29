
```cmd
conda create -n sb3 python=3.10 -y
conda activate sb3

pip install --upgrade pip
pip install stable-baselines3 #Don't add [extra]
pip install gym==0.26.2

# ⚠️SB3는 MuJoCo 설치가 필요 없을 경우 위만 설치해도 충분
# pip install mujoco
# pip install imageio imageio-ffmpeg  # 영상 저장용

```

|항목	|내용|
|--|--|
|환경 이름|	sb3|
|저장 파일	|expert_data.npz|
|SB3 알고리즘|	PPO (가장 무난)|
|환경	|CartPole-v1 (가볍고 빠름)|