import numpy as np
data = np.load("expert_data.npz")
print("obs.shape:", data["obs"].shape)
print("actions.shape:", data["actions"].shape)

#obs.shape: (1000, 4)
#actions.shape: (1000,)