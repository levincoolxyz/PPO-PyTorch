import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

env = gym.make('AntsEnv-v0',Nmax=12)
env.reset()

rrand = []
dt = env.dt
Nsim = 1500

for i in range(Nsim):
    env.render()
    # take a random action and check untrained reward
    obs, reward, done, info = env.step(env.action_space.sample())

    rrand = np.append(rrand,reward)

print('Episode: rand\tReward: {}'.format(int(np.sum(rrand))))

env.reset()

roptm = []
dt = env.dt
Nsim = 1500

for i in range(Nsim):
    env.render()
    # take a random action and check untrained reward
    # obs, reward, done, info = env.step(env.action_space.sample())

    # """
    # check maximum reward
    # optimal strategy
    pull_threshold = np.pi/2.1
    pullProb = 1 - np.logical_or(env.theta < pull_threshold, env.theta > (np.pi*2-pull_threshold))*1
    # print(pullProb)
    position, velocity = env.state
    pullDir = - position[2] - env.theta
    larger = angle_normalize(pullDir) > env.dphi/2
    smller = angle_normalize(pullDir) < -env.dphi/2
    pullDir[larger] = env.dphi/2
    pullDir[smller] = -env.dphi/2
    optimal_act = np.append((pullProb[1:]-.5)*2,pullDir[1:]/env.dphi*2)
    obs, reward, done, info = env.step(optimal_act)
    # """

    roptm = np.append(roptm,reward)

env.close()

print('Episode: opt\tReward: {}'.format(int(np.sum(roptm))))

"""
fig, ax = plt.subplots()
ax.plot(np.linspace(0,Nsim*dt,Nsim), rrand, 'go-')
ax.plot(np.linspace(0,Nsim*dt,Nsim), roptm, 'go-')
ax.grid()
ax.set(xlabel='timestep', ylabel='reward (velocity)', title='Reward History')
plt.show()
"""