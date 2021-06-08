# export DISPLAY=:0 

import sys

import matplotlib.pyplot as plt

sys.path.append("../")

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats, rgb2gray, action_to_id,id_to_action

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def run_episode(env, agent, deterministic, skip_frames=3,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96 )
    state = state[np.newaxis,:,:,:]
    print("state shape", np.shape(state))
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        # Hint: frame skipping might help you to get better results.

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(-1, history_length + 1,96, 96 )

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        #print("p3 state shape", np.shape(state))
        #state = np.reshape(state,(1,96,96))
        #state = state[np.newaxis, :, :, :]
        #print("p4 state shape", np.shape(state))

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"),name = "Train CarRacing",
                             stats= ["episode_reward", "straight", "left", "right", "accel", "brake"])
    training,evaluation = [],[]
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        if (i < 30):
            stats = run_episode(env, agent, history_length=history_length, max_timesteps=100, deterministic=False, do_training=True)
        else:
            stats = run_episode(env, agent, history_length=history_length,deterministic=False, do_training=True)

        training.append(stats.episode_reward)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        # store model.

        if i% eval_cycle == 0:
            for j in range(num_eval_episodes):
                reward = 0
                stats_eval= run_episode(env, agent,history_length=history_length,deterministic=True,do_training=False)
                reward += stats_eval.episode_reward

            reward = reward/ num_eval_episodes # averaging the reward over the number of episodes
            print("i = ",i,"mean episode reward=",reward)
            evaluation.append(reward)

        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "hist"+str(history_length)+"episode"+str(i)+"dqn_agent.ckpt"))

    tensorboard.close_session()
    return training,evaluation

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20

    env = gym.make('CarRacing-v0').unwrapped
    history_length=3
    
    # TODO: Define Q network, target network and DQN agent
    # ...

    net = CNN(history_length=history_length,n_classes=5)
    dqn_agent = DQNAgent(net, net, num_actions=5, history_length=1e6)
    #train_online(env, agent=dqn_agent, num_episodes=200)

    training,evaluation = train_online(env, dqn_agent, num_episodes=800, history_length=history_length, model_dir="./models_carracing")

    plt.plot(training)
    plt.title("Training _hist3 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Training_car.jpg")
    plt.show()

    plt.plot(evaluation)
    plt.title("Eval_hist3 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Eval_car.jpg")
    plt.show()


    history_length=1

    net = CNN(history_length=history_length, n_classes=5)
    dqn_agent = DQNAgent(net, net, num_actions=5, history_length=1e6)
    # train_online(env, agent=dqn_agent, num_episodes=200)

    training, evaluation = train_online(env, dqn_agent, num_episodes=800, history_length=history_length,
                                        model_dir="./models_carracing")

    plt.plot(training)
    plt.title("Training_hist1 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Training_car2.jpg")
    plt.show()

    plt.plot(evaluation)
    plt.title("Eval_hist1 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Eval_car2.jpg")
    plt.show()

    history_length=5

    net = CNN(history_length=history_length, n_classes=5)
    dqn_agent = DQNAgent(net, net, num_actions=5, history_length=1e6)
    # train_online(env, agent=dqn_agent, num_episodes=200)

    training, evaluation = train_online(env, dqn_agent, num_episodes=800, history_length=history_length,
                                        model_dir="./models_carracing")

    plt.plot(training)
    plt.title("Training_hist5 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Training_car3.jpg")
    plt.show()

    plt.plot(evaluation)
    plt.title("Eval_hist5 CarRacing")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("Eval_car3.jpg")
    plt.show()


