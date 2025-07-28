import torchvision
import numpy as np 
import gym
import torch
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style

def to_input (states, actions=None,  n=2, compare=1):
    print("<Func: to_input>")
    '''
    Data preperpation and filtering 
    Inputs:
    states: expert states as tensor
    actions: actions states as tensor
    n: window size (how many states needed to predict the next action)
    compare: for filtering data 
    return:
    output_states: filtered states as tensor 
    output_actions: filtered actions as tensor if actions != None
    '''
    count=0
    index= []

    if type(actions) != torch.Tensor:
        ep, t, state_size = states.shape
    else:
        ep, t, state_size = states.shape
        _, _, action_size = actions.shape

    
    if type(actions) != torch.Tensor:
        output_states = torch.zeros((ep*(t-n+1) , state_size*n), dtype = torch.float)
    else:
        output_states = torch.zeros((ep*(t-n+1) , state_size*n), dtype = torch.float)
        output_actions = torch.zeros((ep*(t-n+1) , action_size), dtype = torch.float)
        
    
    for i in range (ep):
        for j in range (t-n+1):
            if (states[i, j] == -compare*torch.ones(state_size)).all() or (states[i, j+1] == -compare*torch.ones(state_size)).all():
                index.append([i,j])
            else:
                output_states[count] = states[i, j:j+n].view(-1)

            if type(actions) != torch.Tensor:
                count+=1
                # do nothing
            else:
                output_actions[count] = actions[i,j]
                count+=1
   
    if type(actions) != torch.Tensor:
        output_states= output_states[:count]
        return output_states
    else:
        output_states  = output_states[:count]
        output_actions = output_actions[:count]
        return output_states, output_actions

def train_transition (state_space_size, training_set, model, n=2,   batch_size = 256, n_epoch = 50):
    print("<Func: train_transition>")
    '''
    train transition model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set: 
    model: transition model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    return:
    model: trained transition model
    '''
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    loss_list = []
    for itr in range(n_epoch):
        total_loss = 0
        b=0
        for batch in range (0,training_set.shape[0], batch_size):
            data   = training_set  [batch : batch+batch_size , :n*state_space_size]
            y      = training_set [batch : batch+batch_size, n*state_space_size:]
            y_pred = model(data)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss/b))
        display.clear_output(wait=True)
        loss_list.append(total_loss / training_set.shape[0])
    return model

def train_BC (state_space_size, training_set , policy,   batch_size = 256, n_epoch = 50, ):
    print("<Func: train_BC>")
    '''
    train Behavioral Cloning model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set: 
    policy: Behavioral Cloning model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    return:
    policy: trained Behavioral Cloning model
    '''
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001) 
    loss_list = []
    for itr in range(n_epoch):
        total_loss = 0
        b=0
        for batch in range (0,training_set.shape[0], batch_size):
            data   = training_set  [batch : batch+batch_size , :state_space_size]
            y      = training_set [batch : batch+batch_size, state_space_size:]
            y_pred = policy(data)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss/b))
        display.clear_output(wait=True)
        loss_list.append(total_loss / training_set.shape[0])
    return policy

def train(env, training_set, testing_set, criterion, state_trainsition_model):
    print("train")
    # init environment
    state_space_size  = env.observation_space.shape[0]

    # learning_rate = 0.01
    # optimizer = torch.optim.Adam(bc_pendulum.parameters(), lr = learning_rate) 
    n=2
    n_epoch = 100
    batch_size = 256#128
    learning_rate = 0.001
    optimizer = torch.optim.Adam(state_trainsition_model.parameters(), lr = learning_rate) 

    loss_list = []
    test_list = []

    for itr in range (n_epoch):
        total_loss = 0
        b = 0
        for batch in range (0,training_set.shape[0], batch_size):
            x      = training_set  [batch : batch+batch_size , :n*state_space_size]
            y      = training_set  [batch : batch+batch_size , n*state_space_size:]
            y_pred = state_trainsition_model(x)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss / b))
        display.clear_output(wait=True)
        loss_list.append(total_loss / b)
        
        test_list.append(criterion(state_trainsition_model(testing_set[:, :n*state_space_size]), testing_set[:, n*state_space_size: ]).item())

    # plot test loss
    #torch.save(bc_pendulum, "bc_pendulum_n=2") # uncomment to save the model
    plt.plot(test_list, label="test loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()

    p = 42511  # select any point to test the model
    print(state_trainsition_model(testing_set[p, :n*state_space_size]))
    print(testing_set[p, n*state_space_size:])
    criterion(state_trainsition_model(testing_set[p, :n*state_space_size]), testing_set[p, n*state_space_size:] ).item()

def train_bco(bco_pendulum, training_set,  criterion, state_space_size, testing_set):
    print("<Func: train_bco>")
    # Train BCO model 
    loss_list = []
    test_loss = []

    batch_size = 256
    n_epoch = 50

    learning_rate = 0.001
    optimizer = torch.optim.Adam(bco_pendulum.parameters(), lr = learning_rate) 

    for itr in range(n_epoch):
        total_loss = 0
        b=0
        for batch in range (0,training_set.shape[0], batch_size):
            data   = training_set  [batch : batch+batch_size , :state_space_size]
            y      = training_set  [batch : batch+batch_size , state_space_size:]
            y_pred = bco_pendulum(data)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss / b))
        display.clear_output(wait=True)
        loss_list.append(total_loss / b)
        x = testing_set[:, :state_space_size]
        y = testing_set[:,state_space_size:]
        y_pred = bco_pendulum(x)
        test_loss.append(criterion(y_pred, y).item())

    # plot test loss for BCO
    # torch.save(bco_pendulum, "bco_pendulum_n=2") #uncomment to save model
    plt.plot(test_loss, label="Testing Loss BCO")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    p = 112 # select any point to test the model
    print( bco_pendulum(testing_set[p, :state_space_size]))
    print(testing_set[p, state_space_size:])
    criterion(bco_pendulum(testing_set[p, :state_space_size]), testing_set[p, state_space_size:] ).item()

# Train BCO($\alpha$) in Pendulum with interacting with environmet
def train_bco_environmet(env, state_trainsition_model, expert_states, bco_pendulum):
    print("<func: train_bco_environmet>")
    action_space_size = env.action_space.shape[0]
    state_space_size  = env.observation_space.shape[0]
    ################################## parameters ##################################
    n=2 # window size
    n_iterations = 100 # max number of interacting with environment
    n_ep = 25 # number of epoches
    max_steps = 200 # max timesteps per epoch
    gamma = 1.0 # discount factor
    seeds = np.zeros(n_iterations) # random seeds
    target_reward = -300 # stop training when reward > targit_reward
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

        states_from_env  = -5*np.ones((n_ep, max_steps, state_space_size)) # states in
        actions_from_env = -5*np.ones((n_ep, max_steps, action_space_size))
        
        for ep in range (n_ep):
            state, _ = env.reset()
            rewards = []
            R=0
            for t in range (max_steps):
                action = bco_pendulum(torch.tensor(state, dtype=torch.float))
                
                action = np.clip(action.detach().numpy(), -2,2) # clip action to be between (-1, 1)
                
                states_from_env[ep,t]  = state
                actions_from_env[ep,t] = action
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
            display.clear_output(wait=True)
        seed_reward.append(G)
        seed_reward_mean.append(G_mean)
        
        print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward_mean[-1])))
        print("Interacting with environment finished")

            
        if np.mean(seed_reward_mean[-1]) > target_reward:
            torch.save(bco_pendulum, "bco_pendulum={}_BCO({})_best_{}_expert_states".format(n,itr,expert_states.shape[0]))        
            break
        ################################ prepare collected states and actions ##################################
        
        states_from_env = torch.tensor(states_from_env, dtype=torch.float)
        actions_from_env = torch.tensor(actions_from_env, dtype=torch.float)
        states_from_env, actions_from_env = to_input(states_from_env, actions_from_env , n=n, compare=5)
        data_env = torch.cat((states_from_env, actions_from_env), 1).detach().numpy()
        np.random.shuffle(data_env)
        data_env = torch.tensor(data_env)
        print("data_env", data_env.shape)
        
        #################################  Update Transition Model and return infeered expert actions ##################################
        
        state_trainsition_model = train_transition(state_space_size, data_env ,  state_trainsition_model,  n=n )
        infered_expert_actions  = state_trainsition_model( torch.tensor(expert_states) )
        infered_expert_actions = torch.tensor( infered_expert_actions, requires_grad=False )
        print("Updated Transition Model and returned infeered expert actions")
        
        ################################# Update BC model ##################################
        
        expert_data = torch.cat((expert_states, infered_expert_actions),1)
        bco_pendulum = train_BC(state_space_size, expert_data, bco_pendulum, n_epoch =100)
        print(" Updated BC model itra= {}".format(itr))
        print("finished")
        
    print(" Updated BC model itra= {}".format(itr))

# Test BCO($\alpha$) in Pendulum environment with 5 random seeds
def test_bco_environmet(env, bco_pendulum):
    print("<func: test_bco_environmet>")

    ##x = itr
    n_iterations = 5
    n_ep = 1000
    max_steps = 200
    gamma = 1.0 # discount factor
    seeds = [684, 559, 629, 192, 835]


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
                action = bco_pendulum(torch.tensor(state, dtype=torch.float))
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
            display.clear_output(wait=True)
        seed_reward.append(G)
        seed_reward_mean.append(G_mean)
        print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward_mean[-1])))
        print("Interacting with environment finished")
    # np.save("reward_mean_pendulum_n={}_bco({})_expert_states={}".format(n, x , expert_states.shape[0]), seed_reward_mean) #uncomment to save the reward over 5 random seeds
    return seed_reward_mean

def main():
    plt.style.use("ggplot")
    print("PyTorch 버전:", torch.__version__)
    print("CUDA 지원 버전:", torch.version.cuda)     # GPU 지원이 있으면 CUDA 버전 문자열
    print("사용 가능한 GPU 개수:", torch.cuda.device_count())

    # init environment------------------------------------------------------------------------
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")
    action_space_size = env.action_space.shape[0]
    state_space_size  = env.observation_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    n=2

    # Load Expert data (States only for BCO)
    expert_states  = torch.tensor(np.load("states_expert_Pendulum.npy"), dtype=torch.float)
    print("expert_states", expert_states.shape)     

    # Load new data (states and actions for BCO)
    states_new_agent  =  torch.tensor(np.load ("states_Pendulum_exploration.npy")[:1000], dtype= torch.float)
    actions_new_agent =  torch.tensor(np.load ("actions_Pendulum_exploration.npy")[:1000], dtype= torch.float)
    print("states_new_agent",states_new_agent.shape)
    print("actions_new_agent",actions_new_agent.shape)

    # selecting number expert trajectories from expert data
    # number_expert_trajectories = 50
    # a= np.random.randint(expert_states.shape[0] - number_expert_trajectories)
    # print(a)
    # expert_state, expert_action = to_input (expert_states[a : a+number_expert_trajectories], expert_actions[a : a+number_expert_trajectories], n=n,  compare=5)
    # print("expert_state", expert_state.shape)

# 1- Behavioral Cloning from Observation BCO and BCO($\alpha$)---------------------------------------------
    # concatenate expert states and actions, divided into 70% training and 30% testing
    states_new_agent, actions_new_agent = to_input(states_new_agent, actions_new_agent, n=n, compare=5 )
    new_agent_data = torch.cat((states_new_agent , actions_new_agent), 1)
    new_agent_data = new_agent_data.detach().numpy()
    np.random.shuffle(new_agent_data)
    new_agent_data = torch.tensor(new_agent_data[:])
    print("new_agent_data", new_agent_data.shape)
    n_samples = int(new_agent_data.shape[0]*0.7)
    training_set = new_agent_data[:n_samples]
    testing_set = new_agent_data[n_samples:]
    print("training_set", training_set.shape)
    print("testing_set", testing_set.shape)

    # Network arch Behavioral Cloning , loss function and optimizer
    state_trainsition_model = nn.Sequential(
        
        nn.Linear(n*state_space_size, 20),
        nn.ReLU(),
        
        nn.Linear(20, 40),
        nn.ReLU(),
        
        nn.Linear(40, 10),
        nn.ReLU(),
        
        nn.Linear(10, action_space_size)
    )
    criterion = nn.L1Loss() 

    train(env, training_set, testing_set, criterion, state_trainsition_model)

# Load pre-trained Transition Model and predict infered expert actions from expert states only-------------
    expert_states = to_input(expert_states,  actions=None, n=n, compare=5)
    expert_states = expert_states.detach().numpy()
    np.random.shuffle(expert_states)
    expert_states = torch.tensor(expert_states)
    # state_trainsition_model = torch.load("pendulum_transition_model_from_exploration_states_l1_n=2") #
    infered_expert_actions  = state_trainsition_model(expert_states).detach().numpy()
    infered_expert_actions = torch.tensor(infered_expert_actions, requires_grad=False)
    infered_expert_actions = torch.clamp(infered_expert_actions, -2, 2)

    # Policy Model
    new_data = torch.cat((expert_states[:, :state_space_size], infered_expert_actions),1)
    n_samples = int(new_data.shape[0]*0.7)
    training_set = new_data[:n_samples]
    testing_set = new_data[n_samples:]
    print("training_set", training_set.shape)
    print("testing_set", testing_set.shape)

    # Network arch, loss function and optimizer
    bco_pendulum = nn.Sequential(
        nn.Linear(state_space_size,40),
        nn.ReLU(),
        
        nn.Linear(40,60),
        nn.ReLU(),
        
        nn.Linear(60,20),
        nn.ReLU(),
        
        nn.Linear(20,action_space_size),
    )
    criterion = nn.L1Loss()

    train_bco(bco_pendulum, training_set, criterion, state_space_size, testing_set)
    train_bco_environmet(env, state_trainsition_model, expert_states, bco_pendulum)
    seed_reward_mean = test_bco_environmet(env, bco_pendulum)

# BCO--------------------------------------------------------------------------------------------
    seed_reward_mean_bco = np.array(seed_reward_mean)
    mean_bco  = np.mean(seed_reward_mean_bco,axis=0)
    std_bco  = np.std(seed_reward_mean_bco,axis=0)
# Expert
    expert  = np.load("reward_mean_pendulum_expert.npy")
    mean_expert= np.mean(expert,axis=0)
    std_expert = np.std(expert,axis=0)
# Random
    random_mean  = np.load("reward_mean_pendulum_random.npy")
    mean_random= np.mean(random_mean,axis=0)
    std_random  = np.std(random_mean,axis=0)
    # Scaled performance
    def scaled (x, min_value, max_value):
        return (x - min_value) / (max_value - min_value)
    bco_score = scaled( mean_bco[-1]  , mean_random[-1] , mean_expert[-1] )

# Compare BCO VS Expert VS Random
    x = np.arange(1000)

    plt.plot(x, mean_expert, "-", label="Expert")
    plt.fill_between(x, mean_expert+std_expert, mean_expert-std_expert, alpha=0.2)

    plt.plot(x, mean_bco, "-", label="BCO")
    plt.fill_between(x, mean_bco + std_bco, mean_bco - std_bco, alpha=0.2)

    plt.plot(x, mean_random, "-", label="Random")
    plt.fill_between(x, mean_random+std_random, mean_random-std_random, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title("Random VS Expert VS BCO in Pendulum")
    plt.legend()
   # test(testing_set, criterion, state_trainsition_model, expert_states)

if __name__ == '__main__':
    main()





# def test(testing_set, criterion, state_trainsition_model, expert_states):
#     n=2 # window size
#     # 최신 gym은 render_mode를 명시해야 렌더링 가능
#     env = gym.make("Pendulum-v1", render_mode="human")
    
#     action_space_size = env.action_space.shape[0]
#     state_space_size  = env.observation_space.shape[0]

#     # --------------------- TEST -------------------------------
#     p = 42511  # select any point to test the model
#     print( state_trainsition_model(testing_set[p, :n*state_space_size]))
#     print(testing_set[p, n*state_space_size:])
#     criterion(state_trainsition_model(testing_set[p, :n*state_space_size]), testing_set[p, n*state_space_size:] ).item()

    
#     # --------------------- TEST in Evironment -------------------------------
#     ################################## parameters ##################################
#     n_iterations = 100 # max number of interacting with environment
#     n_ep = 1000 # number of epoches
#     max_steps = 200 # max timesteps per epoch
#     gamma = 1.0 # discount factor
#     seeds = np.zeros(n_iterations) # random seeds
#     target_reward = -300 # stop training when reward > targit_reward
#     ################################## parameters ##################################


#     seed_reward_mean = []
#     seed_reward  = []

#     for itr in range (n_iterations):
#     ################################## interact with env ##################################
#         G= []
#         G_mean = []
#         env.reset(seed=int(seeds[itr]))
#         torch.manual_seed(int(seeds[itr]))
#         torch.cuda.manual_seed_all(int(seeds[itr]))

#         states_from_env  = -5*np.ones((n_ep, max_steps, state_space_size)) # states in
#         actions_from_env = -5*np.ones((n_ep, max_steps, action_space_size))
        
#         for ep in range (n_ep):
#             state = env.reset()
#             rewards = []
#             R=0
#             for t in range (max_steps):
#                 action = bco_pendulum(torch.tensor(state, dtype=torch.float))
                
#                 action = np.clip(action.detach().numpy(), -2,2) # clip action to be between (-1, 1)
                
#                 states_from_env[ep,t]  = state
#                 actions_from_env[ep,t] = action
#                 next_state, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
#                 rewards.append(reward)
#                 state = next_state
#                 if done:
#                     break
#             R = sum([rewards[i]*gamma**i for i in range (len(rewards))])
#             G.append(R)
#             G_mean.append(np.mean(G))
            
            
#             if ep % 1 ==0:
#                 print("ep = {} , Mean Reward = {:.6f}".format(ep, R))
#             display.clear_output(wait=True)
#         seed_reward.append(G)
#         seed_reward_mean.append(G_mean)
        
#         print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward_mean[-1])))
#         print("Interacting with environment finished")

            
#         if np.mean(seed_reward_mean[-1]) > target_reward:
#             torch.save(bco_pendulum, "bco_pendulum={}_BCO({})_best_{}_expert_states".format(n,itr,expert_states.shape[0]))        
#             break
#         ################################ prepare collected states and actions ##################################
        
#         states_from_env = torch.tensor(states_from_env, dtype=torch.float)
#         actions_from_env = torch.tensor(actions_from_env, dtype=torch.float)
#         states_from_env, actions_from_env = to_input(states_from_env, actions_from_env , n=n, compare=5)
#         data_env = torch.cat((states_from_env, actions_from_env), 1).detach().numpy()
#         np.random.shuffle(data_env)
#         data_env = torch.tensor(data_env)
#         print("data_env", data_env.shape)
        
#         #################################  Update Transition Model and return infeered expert actions ##################################
        
#         state_trainsition_model = train_transition( data_env ,  state_trainsition_model,  n=n )
#         infered_expert_actions  = state_trainsition_model( torch.tensor(expert_states) )
#         infered_expert_actions = torch.tensor( infered_expert_actions, requires_grad=False )
#         print("Updated Transition Model and returned infeered expert actions")
        
#         ################################# Update BC model ##################################
        
#         expert_data = torch.cat((expert_states, infered_expert_actions),1)
#         bco_pendulum = train_BC(expert_data, bco_pendulum, n_epoch =100)
#         print(" Updated BC model itra= {}".format(itr))
#         print("finished")
        
#     print(" Updated BC model itra= {}".format(itr))
#     # np.save("reward_mean_pendulum_bc_expert_states={}".format(new_data.shape[0]), seed_reward_mean) #uncomment to save reward over 5 random seeds
