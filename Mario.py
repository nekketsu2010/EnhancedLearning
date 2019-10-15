import time

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
from rl.core import Processor


my_action = [
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

env = JoypadSpace(env, my_action)

nb_actions = 3
window_length = 1
input_shape = (window_length,) + env.observation_space.shape
print(input_shape)

# ゲーム環境のリセット
env.reset()

from keras.models import Sequential
from keras.layers import *
from keras.initializers import he_normal
model = Sequential()
print('input_shape' + str(input_shape))
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions, activation='softmax', kernel_initializer=he_normal()))

print(model.summary())

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam

memory = SequentialMemory(limit=3000, window_length=window_length)
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory)
                 #nb_steps_warmup=10, target_model_update=1e-2, policy=policy, enable_double_dqn=True)
agent.compile(Adam())

# fit の結果を取得しておく
history = agent.fit(env, nb_steps=50_000, visualize=False, verbose=1)

agent.test(env, nb_episodes=80000, visualize=True)
