from __future__ import print_function

import sys

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=0):
    episode_reward = 0
    step = 0

    state = env.reset()
    first_time = True

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    while True:

        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...

        # print(type(state))
        state = rgb2gray(state)
        state = state[np.newaxis, np.newaxis, :, :]
        s_new = state
        if (history_length > 0):
            s_new = np.zeros((len(state), 1 + history_length, 96, 96))
            # print("x_train_new shape=",np.shape(x_train_new))
            for i, img in enumerate(state):
                temp = state[i]
                for n in range(history_length):
                    # for the intial images where sufficient history is not present,
                    # I attach the same scene as input to the network
                    # print("X_train shape", np.shape(X_train[i]),"n=", n)
                    temp = np.concatenate((temp, state[i]), axis=0)
                s_new[i] = temp
        # print(np.shape(s_new))

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...

        output = agent.predict(s_new)
        output = torch.unsqueeze(output, 0)  # Tensor[5] --> Tensor[1,5]
        _, predicted = torch.max(output.data, 1)

        a = id_to_action(predicted) #recheck..

        #print("Pred=", predicted, "a=", a)
        if first_time == True:
            next_state, r, done, info = env.step([0.0, 1, 0.0])
            first_time=False
        else:
            next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test
    history_length = 0

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")
    agent = BCAgent(history_length=history_length)
    agent.load("models/agent_0hist_800runs_wo_upsampl.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, history_length=history_length)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    print(results)
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
