import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

env = gym.make('AntsEnv-v0',Nmax=12)

# untangle observation/action space order
index = list(range((env.Nmax-1)*2))
inverse = np.append(index[:-1:2],index[1::2])
ants = np.array(range(env.Nmax-1))
order = np.concatenate((ants[:,None],ants[:,None]+env.Nmax-1),axis=1).flatten()

'''
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
'''

env.reset()
rreal = []
dt = env.dt
Nsim = 1500

for i in range(Nsim):
    env.render()
    # statistically 'realistic' strategy (do not use states)
    if i == 0:
        obs, reward, done, info = env.step(env.action_space.sample())
    else:
        obs = obs[inverse]
        dotProd = obs[:env.Nmax-1]*np.cos(obs[env.Nmax-1:]) # force threshold assumed to be 0
        pullDir = obs[env.Nmax-1:]
        larger = angle_normalize(pullDir) > env.dphi/2
        smller = angle_normalize(pullDir) < -env.dphi/2
        pullDir[larger] = env.dphi/2
        pullDir[smller] = -env.dphi/2
        optimal_act = np.append(np.tanh(dotProd),pullDir/env.dphi*2)
        obs, reward, done, info = env.step(optimal_act[order])

    rreal = np.append(rreal,reward)

env.close()

print('Episode: real\tReward: {}'.format(int(np.sum(rreal))))

env.reset()
rbest = []
dt = env.dt
Nsim = 1500

for i in range(Nsim):
    env.render()

    # best possible strategy (ignores observations, uses states)
    pull_threshold = np.pi/2
    pullProb = 1 - np.logical_or(env.theta < pull_threshold, env.theta > (np.pi*2-pull_threshold))*1
    position, velocity = env.state
    pullDir = - position[2] - env.theta
    larger = angle_normalize(pullDir) > env.dphi/2
    smller = angle_normalize(pullDir) < -env.dphi/2
    pullDir[larger] = env.dphi/2
    pullDir[smller] = -env.dphi/2
    best_act = np.append((pullProb[1:]-.5)*2,pullDir[1:]/env.dphi*2)
    obs, reward, done, info = env.step(best_act[order])

    rbest = np.append(rbest,reward)

env.close()

print('Episode: best\tReward: {}'.format(int(np.sum(rbest))))

"""
fig, ax = plt.subplots()
ax.plot(np.linspace(0,Nsim*dt,Nsim), rrand, 'go-')
ax.plot(np.linspace(0,Nsim*dt,Nsim), rreal, 'go-')
ax.plot(np.linspace(0,Nsim*dt,Nsim), rbest, 'go-')
ax.grid()
ax.set(xlabel='timestep', ylabel='reward (velocity)', title='Reward History')
plt.show()
"""