import gym
from gym import wrappers
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# env = gym.make('Seastar-v0',dt=0.1)
env = gym.make('Seastar-v0',Nfeet=10,Lbody=5,dt=0.1)
Nsim = 500
dt = env.dt

#env = wrappers.Monitor(env, './results/rand/' + str(time.time()) + '/')
# obs = env.reset(rand=False)
obs = env.reset()
rrand = []

for i in range(Nsim):
    env.render()
    # action = env.action_space.sample() # random
    action = np.array((env.l - 2>0)*1) # simple design
    # action = np.array((env.l>0)*1) # exploit 1
    # _, theta = env.state
    # action = np.array((theta>0)*1) # exploit 2
    obs, reward, done, info = env.step(action)
    if done:
        Nsim=i
        break

    rrand = np.append(rrand,reward)

env.close()
print('Episode: rand\tReward: {}'.format(int(np.sum(rrand))))

"""
fig, ax = plt.subplots()
ax.plot(np.linspace(0,Nsim*dt,Nsim), rrand, 'go-')
ax.grid()
ax.set(xlabel='timestep', ylabel='reward (velocity)', title='Reward History')
plt.show()
"""