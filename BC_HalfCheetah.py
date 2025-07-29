import numpy as np 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style

def to_input (states, actions,  n=2, compare=1):
    '''
    Data preperpation and filtering 
    Inputs:
    states: expert states as tensor
    actions: actions states as tensor
    n: window size (how many states needed to predict the next action)
    compare: for filtering data 
    return:
    output_states: filtered states as tensor 
    output_actions: filtered actions as tensor 
    '''
    count=0
    index= []
    ep, t, state_size = states.shape
    _, _, action_size = actions.shape
    
    output_states = torch.zeros((ep*(t-n+1) , state_size*n), dtype = torch.float64)
    output_actions = torch.zeros((ep*(t-n+1) , action_size), dtype = torch.float32)
    
    for i in range (ep):
        for j in range (t-n+1):
            if (states[i, j] == -compare*torch.ones(state_size)).all() or (states[i, j+1] == -compare*torch.ones(state_size)).all():
                index.append([i,j])
            else:
                output_states[count] = states[i, j:j+n].view(-1)
                output_actions[count] = actions[i,j]
                count+=1
    output_states= output_states[:count]
    output_actions= output_actions[:count]
    
    return output_states, output_actions

def train(env, bc_halfcheetah, training_set, testing_set, criterion):
    # init environment
    state_space_size  = env.observation_space.shape[0]

    loss_list = []
    test_loss = []
    batch_size = 256
    n_epoch = 50
    learning_rate = 0.001
    optimizer = torch.optim.Adam(bc_halfcheetah.parameters(), lr = learning_rate) 
    for itr in range(n_epoch):
        total_loss = 0
        b=0
        for batch in range (0,training_set.shape[0], batch_size):
            data   = training_set  [batch : batch+batch_size , :state_space_size]
            y      = training_set [batch : batch+batch_size, state_space_size:]
            y_pred = bc_halfcheetah(data)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr+1, total_loss / b))
        display.clear_output(wait=True)
        loss_list.append(total_loss / b)
        x = testing_set[:, :state_space_size]
        y = testing_set[:,state_space_size:]
        y_pred = bc_halfcheetah(x)
        test_loss.append(criterion(y_pred, y).item())

    # plot test loss
    torch.save(bc_halfcheetah, "saves/bc_halfcheetah_n=2") # uncomment to save the model
    plt.plot(test_loss, label="Testing Loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def test(bc_halfcheetah, testing_set, criterion):
    # 최신 gym은 render_mode를 명시해야 렌더링 가능
    env = gym.make("HalfCheetah-v5", render_mode="human")
    
    state_space_size  = env.observation_space.shape[0]

    # --------------------- TEST -------------------------------
    p = 87 # select any point to test the model
    print(bc_halfcheetah(testing_set[p, :state_space_size]) )
    print(testing_set[p, state_space_size:])
    criterion( bc_halfcheetah(testing_set[p, :state_space_size] ), testing_set[p, state_space_size:] ).item()

    # --------------------- TEST in Evironment -------------------------------
    ################################## parameters ##################################
    # n=2 # window size
    n_iterations = 5 # max number of interacting with environment
    n_ep = 200 #1000 # number of epoches
    max_steps = 500 # max timesteps per epoch
    gamma = 1.0 # discount factor
    seeds = [684, 559, 629, 192, 835] # random seeds for testing
    ################################## parameters ##################################

    seed_reward_mean = []
    seed_reward  = []
    for itr in range (n_iterations):
    ################################## interact with env ##################################
        G= []
        G_mean = []
        env.reset(seed=int(seeds[itr]))
        torch.manual_seed(int(seeds[itr]))
        torch.cuda.manual_seed_all(int(seeds[itr]))

        for ep in range (n_ep):
            state, _ = env.reset()
            rewards = []
            R=0
            for t in range (max_steps):      
                action = bc_halfcheetah(torch.tensor(state, dtype=torch.float))
                action = np.clip(action.detach().numpy(), -2,2)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards.append(reward)
                state = next_state
                if done:
                    break
            R = sum([rewards[i]*gamma**i for i in range (len(rewards))])
            G.append(R)
            G_mean.append(np.mean(G))
            if ep % 1 ==0:
                print("ep = {} , Mean Reward = {:.6f}".format(ep, R))
            #display.clear_output(wait=True)
        seed_reward.append(G)
        seed_reward_mean.append(G_mean)
        print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward_mean[-1])))
        print("Interacting with environment finished")
    env.close()
    return seed_reward_mean

def main():
    plt.style.use("ggplot")

    env_name='HalfCheetah-v5'
    env = gym.make(env_name)
    action_space_size = env.action_space.shape[0]
    state_space_size  = env.observation_space.shape[0]

    # Load Expert data (states and actions for BC, States only for BCO)
    data = np.load("expert_data_halfcheetah.npz")
    expert_states  = torch.tensor(data["obs"], dtype=torch.float64)
    expert_actions = torch.tensor(data["actions"], dtype=torch.float32)
    print("expert_states", expert_states.shape)
    print("expert_actions", expert_actions.shape)

    # selecting number expert trajectories from expert data
    number_expert_trajectories = 50
    a= np.random.randint(expert_states.shape[0] - number_expert_trajectories) #500(trajectories)-50
    print(a)
    expert_state, expert_action = to_input (expert_states[a : a+number_expert_trajectories], expert_actions[a : a+number_expert_trajectories], n=2,  compare=5)
    print("expert_state", expert_state.shape)
    print("expert_action", expert_action.shape)

    # concatenate expert states and actions, divided into 70% training and 30% testing
    new_data = np.concatenate((expert_state[:,: state_space_size], expert_action), axis=1)
    np.random.shuffle(new_data)
    new_data = torch.tensor(new_data, dtype=torch.float)
    n_samples = int(new_data.shape[0]*0.7)
    training_set = new_data[:n_samples]
    testing_set = new_data[n_samples:]
    print("training_set", training_set.shape)
    print("testing_set", testing_set.shape)

    # Network arch Behavioral Cloning , loss function and optimizer
    bc_halfcheetah =  nn.Sequential(
        nn.Linear(state_space_size,40),
        nn.ReLU(),
        
        nn.Linear(40,80),
        nn.ReLU(),
        
        nn.Linear(80,120),
        nn.ReLU(),
        
        nn.Linear(120,100),
        nn.ReLU(),
        
        nn.Linear(100,40),
        nn.ReLU(),
        
        nn.Linear(40,20),
        nn.ReLU(),
        
        
        nn.Linear(20,action_space_size),
    )

    criterion = nn.MSELoss()

    train(env, bc_halfcheetah, training_set, testing_set, criterion)
    seed_reward_mean = test(bc_halfcheetah, testing_set, criterion)
    #compare(seed_reward_mean)

if __name__ == '__main__':
    main()