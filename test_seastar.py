import gym
from gym import wrappers
import time
from PPO_seastar import PPO, Memory
from PIL import Image
import torch

def test():
    ############## Hyperparameters ##############
    env_name = "Seastar-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    max_timesteps = 500
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 10
    max_timesteps = 300
    render = True
    save_gif = False

    # filename = "PPO_{}.pth".format(env_name)
    # filename = "PPO_{}_1.pth".format(env_name)
    filename = "PPO_{}_4.pth".format(env_name)
    # directory = "./preTrained/"
    directory = "./"
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            else:
                own_state[name].copy_(param)
        return self

    preTrainedParam = torch.load(directory+filename)
    preTrainedParam.pop('affine.weight',None)
    preTrainedParam.pop('affine.bias',None)
    preTrainedParam.pop('value_layer.0.weight',None)
    preTrainedParam.pop('value_layer.2.weight',None)
    preTrainedParam.pop('value_layer.4.weight',None)
    preTrainedParam.pop('value_layer.0.bias',None)
    preTrainedParam.pop('value_layer.2.bias',None)
    preTrainedParam.pop('value_layer.4.bias',None)
    load_my_state_dict(ppo.policy_old,preTrainedParam)
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        env = wrappers.Monitor(env, './results/seastar/' + str(time.time()) + '/')
        # state = env.reset(rand=False)
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            print(action)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
    
if __name__ == '__main__':
    test()
    
    
