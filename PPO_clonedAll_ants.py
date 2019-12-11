import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np

deviceName = "cuda:0" if torch.cuda.is_available() else "cpu"
deviceName = "cpu"
device = torch.device(deviceName)

class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1 (network of an *individual ant* 2 in 2 out)
        self.ant =  nn.Sequential(
                nn.Linear(2, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 2),
                nn.Tanh()
                )

        # critic
        self.val = nn.Sequential(
                nn.Linear(2, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def actor(self, observations):
        out = [self.ant(observations[:,i:i+2]) for i in range(0,list(observations.size())[1],2)]
        return torch.cat(out,dim=1)

    def critic(self, observations):
        out = [self.val(observations[:,i:i+2]) for i in range(0,list(observations.size())[1],2)]
        return torch.mean(torch.cat(out,dim=1),dim=1)

    def act(self, observations, memory):
        action_mean = self.actor(observations)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.observations.append(observations)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, observations, action):   
        action_mean = torch.squeeze(self.actor(observations))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        observation_value = self.critic(observations)
        
        return action_logprobs, torch.squeeze(observation_value), dist_entropy

class PPO:
    def __init__(self, observation_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(observation_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(observation_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, observations, memory):
        observations = torch.FloatTensor(observations.reshape(1, -1)).to(device)
        return self.policy_old.act(observations, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_observations = torch.squeeze(torch.stack(memory.observations).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, value_estimate, dist_entropy = self.policy.evaluate(old_observations, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - value_estimate.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(value_estimate, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "AntsEnv-v0"
    render = False
    solved_reward = 600         # stop training if avg_reward > solved_reward (actual optimal ~ 1630)
    log_interval = 20           # print avg reward in the interval
    max_episodes = 100000       # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    # random_seed = None
    random_seed = 1
    #############################################
    
    # creating environment
    # Nants = 20
    Nants = 8
    env = gym.make(env_name,Nmax=Nants,dphi=50)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(observation_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    filename = "PPO_cloned_{}_{}.pth".format(env_name,deviceName)
    directory = "./preTrained/"
    preTrainedParam = torch.load(directory+filename, map_location=device)
    preTrainedParam.pop('critic.0.weight',None)
    preTrainedParam.pop('critic.2.weight',None)
    preTrainedParam.pop('critic.4.weight',None)
    preTrainedParam.pop('critic.0.bias',None)
    preTrainedParam.pop('critic.2.bias',None)
    preTrainedParam.pop('critic.4.bias',None)
    load_existing_param(ppo.policy,preTrainedParam)
    load_existing_param(ppo.policy_old,preTrainedParam)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        observations = env.reset(rand=True)
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(observations, memory)
            observations, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)

                obs_hist = torch.squeeze(torch.stack(memory.observations).to(device)).detach()
                torch.save(obs_hist, './Obs_clonedAll_{}_{}_{}.pt'.format(env_name,deviceName,time_step))

                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_clonedAll_solved_{}_{}.pth'.format(env_name,deviceName))
            break
        
        # save every 100 episodes
        if i_episode % 100 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_clonedAll_{}_{}.pth'.format(env_name,deviceName))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length/log_interval
            running_reward = running_reward/log_interval
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
    
def load_existing_param(network, state_dict):

    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        else:
            own_state[name].copy_(param)
    return network

if __name__ == '__main__':
    main()