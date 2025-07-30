import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

n_envs = 16

# Parallel environments
vec_env = make_vec_env("FrozenLake-v1", n_envs=n_envs)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=50000, progress_bar=True)
model.save("ppo_test_FrozenLake")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_test_FrozenLake")

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

# success rate = 0.03446601941747573 (710.0/20600)
# success rate = 0.03434782608695652 (711.0/20700)
# success rate = 0.034423076923076924 (716.0/20800)
# success rate = 0.034545454545454546 (722.0/20900)
# success rate = 0.034523809523809526 (725.0/21000)
# success rate = 0.03469194312796209 (732.0/21100)
# success rate = 0.03471698113207547 (736.0/21200)
# success rate = 0.03464788732394366 (738.0/21300)
# success rate = 0.03453271028037383 (739.0/21400)
# success rate = 0.034678899082568805 (756.0/21800)
# success rate = 0.03454954954954955 (767.0/22200)
# success rate = 0.034484304932735424 (769.0/22300)
# success rate = 0.03448888888888889 (776.0/22500)
# success rate = 0.034380530973451326 (777.0/22600)
# success rate = 0.03440528634361233 (781.0/22700)
# success rate = 0.0343859649122807 (784.0/22800)
# success rate = 0.03423580786026201 (784.0/22900)
# success rate = 0.034155844155844155 (789.0/23100)
# success rate = 0.03405172413793103 (790.0/23200)
# success rate = 0.03407725321888412 (794.0/23300)
# success rate = 0.03418803418803419 (800.0/23400)
# success rate = 0.03412765957446808 (802.0/23500)
# success rate = 0.034184100418410045 (817.0/23900)
# success rate = 0.034125 (819.0/24000)
# success rate = 0.034173553719008265 (827.0/24200)
# success rate = 0.034221311475409834 (835.0/24400)
# success rate = 0.03422764227642276 (842.0/24600)
# success rate = 0.03425101214574899 (846.0/24700)
# success rate = 0.034193548387096775 (848.0/24800)
# success rate = 0.0342 (855.0/25000)
# success rate = 0.03418326693227092 (858.0/25100)
# success rate = 0.034229249011857706 (866.0/25300)
# success rate = 0.0341015625 (873.0/25600)
# success rate = 0.03401544401544401 (881.0/25900)
# success rate = 0.034 (884.0/26000)
# success rate = 0.03393129770992366 (889.0/26200)
# success rate = 0.03395437262357415 (893.0/26300)
# success rate = 0.03386363636363637 (894.0/26400)
# success rate = 0.0340377358490566 (902.0/26500)
# success rate = 0.03408239700374532 (910.0/26700)
# success rate = 0.0341044776119403 (914.0/26800)
# success rate = 0.034317343173431734 (930.0/27100)
# success rate = 0.03433823529411765 (934.0/27200)
# success rate = 0.0341970802919708 (937.0/27400)
# success rate = 0.034202898550724635 (944.0/27600)
# success rate = 0.03414285714285714 (956.0/28000)
# success rate = 0.03402826855123675 (963.0/28300)
# success rate = 0.03397887323943662 (965.0/28400)
# success rate = 0.03391608391608392 (970.0/28600)
# success rate = 0.033902439024390246 (973.0/28700)
# success rate = 0.03409722222222222 (982.0/28800)
# success rate = 0.034083044982698964 (985.0/28900)
# success rate = 0.034 (986.0/29000)
# success rate = 0.03424657534246575 (1000.0/29200)
# success rate = 0.034368600682593856 (1007.0/29300)
# success rate = 0.03427118644067797 (1011.0/29500)