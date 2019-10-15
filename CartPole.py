import gym
env = gym.make('Pendulum-v0')

window_length = 1
input_shape = (window_length,) + env.observation_space.shape

nb_actions = env.action_space.n

from keras.models import Sequential
from keras.layers import *
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam

memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
agent.compile(Adam())

# fit の結果を取得しておく
history = agent.fit(env, nb_steps=10000, visualize=True, verbose=1)

agent.test(env, nb_episodes=5, visualize=True)

import matplotlib.pyplot as plt


# 結果を表示
plt.subplot(2,1,1)
plt.plot(history.history["nb_episode_steps"])
plt.ylabel("step")

plt.subplot(2,1,2)
plt.plot(history.history["episode_reward"])
plt.xlabel("episode")
plt.ylabel("reward")

plt.show()  # windowが表示されます。