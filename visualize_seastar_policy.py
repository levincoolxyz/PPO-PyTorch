import gym
from PPO_seastar import PPO, Memory
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_policy():
    ############## Hyperparameters ##############
    env_name = "Seastar-v0"
    sensor = "ltxy"
    # sensor = "l"
    # creating environment
    env = gym.make(env_name,SensorMode=sensor)
    state_dim = env.observation_space.shape[0]
    action_dim = env.Nfeet*2

    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    # filename = "PPO_{}.pth".format(env_name)
    filename = "PPO_{}_{}.pth".format(env_name,sensor)
    # directory = "./preTrained/"
    directory = "./"

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename,map_location=device))
    
    # Nres = 1000
    # test_lengths = torch.linspace(0,3,Nres).reshape(1,-1).to(device)
    # detach_likelihood = torch.zeros(Nres).reshape(1,-1).to(device)
    # for i in range(Nres):
    #     detach_likelihood[:,i]=ppo.policy_old.action_sublayer(test_lengths[:,i])[1]
    # 
    # fig, ax = plt.subplots()
    # ax.plot(test_lengths.data.cpu().numpy(), detach_likelihood, 'ko-')
    # ax.grid()
    # ax.set(xlabel='foot length', ylabel='detach probability', title='Trained Policy')
    # plt.show()

    Nres = 100
    test_lengths = np.linspace(0,3,Nres)
    test_thetas = np.linspace(-np.pi/2,np.pi/2,Nres)
    test_xdots = np.linspace(-2,2,Nres)
    test_ydots = np.linspace(-2,2,Nres)

    meshl, mesht = np.meshgrid(test_lengths,test_thetas)
    meshl, meshx = np.meshgrid(test_lengths,test_xdots)
    meshl, meshy = np.meshgrid(test_lengths,test_ydots)
    detach1 = np.zeros((Nres,Nres))
    detach2 = np.zeros((Nres,Nres))
    detach3 = np.zeros((Nres,Nres))
    for i in range(Nres):
        for j in range(Nres):
            input1 = torch.tensor([meshl[j,i], mesht[j,i], 0, 0]).reshape(1,-1).to(device)
            input2 = torch.tensor([meshl[j,i], 0, meshx[j,i], 0]).reshape(1,-1).to(device)
            input3 = torch.tensor([meshl[j,i], 0, 0, meshy[j,i]]).reshape(1,-1).to(device)

            detach1[j,i]=ppo.policy_old.action_sublayer(input1)[:,1].data.cpu().numpy()
            detach2[j,i]=ppo.policy_old.action_sublayer(input2)[:,1].data.cpu().numpy()
            detach3[j,i]=ppo.policy_old.action_sublayer(input3)[:,1].data.cpu().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1,projection='3d')
    ax1.plot_surface(meshl, mesht, detach1, cmap='viridis', edgecolor='none')
    ax1.set(xlabel='foot length', ylabel='foot angle', title='detach probability')

    ax2 = fig.add_subplot(1,3,2,projection='3d')
    ax2.plot_surface(meshl, meshx, detach2, cmap='viridis', edgecolor='none')
    ax2.set(xlabel='foot length', ylabel='foot xdot', title='detach probability')

    ax3 = fig.add_subplot(1,3,3,projection='3d')
    ax3.plot_surface(meshl, meshy, detach3, cmap='viridis', edgecolor='none')
    ax3.set(xlabel='foot length', ylabel='foot ydot', title='detach probability')
    
    plt.show()
    
if __name__ == '__main__':
    visualize_policy()
    
    
