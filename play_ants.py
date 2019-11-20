import gym
from gym import wrappers
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

env = gym.make('AntsEnv-v0', Nmax=12, dphi=90)
Nsim = 500
dt = env.dt

# untangle observation/action space order
index = list(range((env.Nmax-1)*2))
inverse = np.append(index[:-1:2],index[1::2])
ants = np.array(range(env.Nmax-1))
order = np.concatenate((ants[:,None],ants[:,None]+env.Nmax-1),axis=1).flatten()

env = wrappers.Monitor(env, './results/rand/' + str(time.time()) + '/')
env.reset(rand=True)
rrand = []

for i in range(Nsim):
    env.render()
    # take a random action and check untrained reward
    obs, reward, done, info = env.step(env.action_space.sample())

    rrand = np.append(rrand,reward)

env.close()
print('Episode: rand\tReward: {}'.format(np.sum(rrand)))

env = wrappers.Monitor(env, './results/null/' + str(time.time()) + '/')
env.reset(rand=True)
rlift = []

for i in range(Nsim):
    env.render()
    # always lift, never pull
    lazy = np.append(np.full(env.Nmax-1,1),np.full(env.Nmax-1,0))
    obs, reward, done, info = env.step(lazy[order])

    rlift = np.append(rlift,reward)

env.close()
print('Episode: lift\tReward: {}'.format(np.sum(rlift)))

env = wrappers.Monitor(env, './results/real/' + str(time.time()) + '/')
obs = env.reset(rand=True)
rreal = []

pull_coeff = 2
pull_threshold = .5

for i in range(Nsim):
    env.render()
    # statistically 'realistic' strategy (do not use states)
    obs = obs[inverse]
    dotProd = obs[:env.Nmax-1]*np.cos(obs[env.Nmax-1:]*np.pi)
    liftProb = 1 - (np.tanh(pull_coeff*(dotProd - pull_threshold))/2 + .5)
    pullDir = np.clip(angle_normalize(-obs[env.Nmax-1:]*np.pi),-env.dphi/2,env.dphi/2)
    real_act = np.append((liftProb-.5)*2,pullDir/env.dphi*2)
    obs, reward, done, info = env.step(real_act[order])

    rreal = np.append(rreal,reward)

env.close()
print('Episode: real\tReward: {}'.format(np.sum(rreal)))

env = wrappers.Monitor(env, './results/best/' + str(time.time()) + '/')
env.reset(rand=True)
rbest = []

for i in range(Nsim):
    env.render()

    # best possible strategy (ignores observations, all ants informed and follow)
    pull_threshold = np.pi/2
    liftProb = 1 - (np.abs(angle_normalize(env.theta - env.theta[0] - env.goalDir)) < pull_threshold)
    position, velocity = env.state
    pullDir = np.clip(angle_normalize(-position[2] - env.theta + env.goalDir),-env.dphi/2,env.dphi/2)
    best_act = np.append((liftProb[1:]-.5)*2,pullDir[1:]/env.dphi*2)
    obs, reward, done, info = env.step(best_act[order])

    rbest = np.append(rbest,reward)

env.close()
print('Episode: best\tReward: {}'.format(np.sum(rbest)))

"""
fig, ax = plt.subplots()
ax.plot(np.linspace(0,Nsim*dt,Nsim), rrand, 'go-')
ax.plot(np.linspace(0,Nsim*dt,Nsim), rreal, 'go-')
ax.plot(np.linspace(0,Nsim*dt,Nsim), rbest, 'go-')
ax.grid()
ax.set(xlabel='timestep', ylabel='reward (velocity)', title='Reward History')
plt.show()
"""