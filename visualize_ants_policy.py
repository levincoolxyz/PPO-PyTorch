import gym
from PPO_cloned_ants import PPO, Memory
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_policy():
    ############## Hyperparameters ##############
    env_name = "AntsEnv-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    # filename and directory to load model from
    # filename = "PPO_cloned_solved_" +env_name+ ".pth"
    filename = "PPO_cloned_" +env_name+ ".pth"
    # directory = "./preTrained/"
    directory = "./"

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename,map_location=device))
    
    Nres = 100
    test_forces = np.linspace(0,1.5,Nres)
    test_thetas = np.linspace(-np.pi,np.pi,Nres)

    meshf, mesht = np.meshgrid(test_forces,test_thetas)
    meshx, meshy = meshf*np.cos(mesht), meshf*np.sin(mesht)

    pull = np.zeros((Nres,Nres))
    phi = np.zeros((Nres,Nres))

    pullO = np.zeros((Nres,Nres))
    phiO = np.zeros((Nres,Nres))

    pull_threshold = 0.9

    for i in range(Nres):
        for j in range(Nres):
            input1 = torch.tensor([meshf[j,i], mesht[j,i]/np.pi]).reshape(1,-1).to(device)
            output1 = ppo.policy_old.ant(input1)

            dotProd = meshf[j,i]*np.cos(mesht[j,i])
            pullO[j,i] = np.tanh(dotProd - pull_threshold)/2.+.5
            phiO[j,i] = np.clip(-mesht[j,i],-env.dphi/2,env.dphi/2)

            pull[j,i] = 1 - (output1[:,0].data.cpu().numpy()/2.+.5)
            phi[j,i] = output1[:,1].data.cpu().numpy()*env.dphi/2.

    from fractions import Fraction
    rat = Fraction(env.dphi/2/np.pi).limit_denominator(100)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='polar')
    cf1 = ax1.contourf(mesht, meshf, pull, np.linspace(0,1), cmap='viridis')
    cb1 = fig.colorbar(cf1,)
    cb1.set_ticks(np.linspace(0,1,5))
    ax1.set(xlabel='x', ylabel='y', title='pull probability')

    ax2 = fig.add_subplot(1,2,2,projection='polar')
    cf2 = ax2.contourf(mesht, meshf, phi, np.linspace(-env.dphi/2, env.dphi/2), cmap='viridis', vmin=-env.dphi/2, vmax=env.dphi/2)
    cb2 = fig.colorbar(cf2,)
    cb2.set_ticks(np.linspace(-env.dphi/2,env.dphi/2,3))
    cb2.ax.set_yticklabels([r'-{}$\pi$'.format(rat),'0',r'{}$\pi$'.format(rat)])
    ax2.set(xlabel='x', ylabel='y', title='pull angle')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='polar')
    cf1 = ax1.contourf(mesht, meshf, pullO, np.linspace(0,1), cmap='viridis')
    cb1 = fig.colorbar(cf1,)
    cb1.set_ticks(np.linspace(0,1,5))
    ax1.set(xlabel='x', ylabel='y', title='pull probability')

    ax2 = fig.add_subplot(1,2,2,projection='polar')
    cf2 = ax2.contourf(mesht, meshf, phiO, np.linspace(-env.dphi/2, env.dphi/2), cmap='viridis', vmin=-env.dphi/2, vmax=env.dphi/2)
    cb2 = fig.colorbar(cf2,)
    cb2.set_ticks(np.linspace(-env.dphi/2,env.dphi/2,3))
    cb2.ax.set_yticklabels([r'-{}$\pi$'.format(rat),'0',r'{}$\pi$'.format(rat)])
    ax2.set(xlabel='x', ylabel='y', title='pull angle')

    plt.show()
    
if __name__ == '__main__':
    visualize_policy()
    
    
