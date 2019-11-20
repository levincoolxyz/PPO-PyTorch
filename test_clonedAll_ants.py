import gym
from gym import wrappers
import time
from PPO_clonedAll_ants import PPO, Memory
from PIL import Image
import torch
import numpy as np

deviceName = "cuda:0" if torch.cuda.is_available() else "cpu"
deviceName = "cpu"
device = torch.device(deviceName)

def test():
    ############## Hyperparameters ##############
    env_name = "AntsEnv-v0"
    # Nants = 20
    Nants = 12
    # Nants = 6
    goalDir = 0
    env = gym.make(env_name,Nmax=Nants,goalDir=goalDir)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 3          # num of episodes to run
    max_timesteps = 500     # max timesteps in one episode
    render = True           # render the environment
    save_gif = False        # png images are saved in gif folder
    
    # filename and directory to load model from
    deviceName = "cpu"
    # filename = "PPO_clonedAll_solved_{}.pth".format(env_name)
    # filename = "PPO_clonedAll_{}.pth".format(env_name)
    filename = "PPO_clonedAll_{}_{}.pth".format(env_name,deviceName)
    # directory = "./preTrained/"
    directory = "./"

    action_std = 0.01       # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    preTrainedParam = torch.load(directory+filename, map_location=device)
    preTrainedParam.pop('critic.0.weight',None)
    preTrainedParam.pop('critic.2.weight',None)
    preTrainedParam.pop('critic.4.weight',None)
    preTrainedParam.pop('critic.0.bias',None)
    preTrainedParam.pop('critic.2.bias',None)
    preTrainedParam.pop('critic.4.bias',None)
    load_existing_param(ppo.policy_old,preTrainedParam)
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        env = wrappers.Monitor(env, './results/cloned/' + str(time.time()) + '/')
        # observation = env.reset()
        observation = env.reset(rand=True)
        for t in range(max_timesteps):
            action = ppo.select_action(observation, memory)
            observation, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, ep_reward))
        ep_reward = 0
        env.close()
    
def load_existing_param(network, state_dict):

    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        else:
            own_state[name].copy_(param)
    return network
    
if __name__ == '__main__':
    test()
    
    