from __future__ import print_function

import sys

sys.path.append("../")

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from sklearn.utils import shuffle, resample


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)
    print("data imported")

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]

    print("X_train shape", np.shape(X_train), "y_train shape=", np.shape(y_train))
    return X_train, y_train, X_valid, y_valid


def upsampling2(X_train, y_train_new):
    # an ugly upsampling technique
    #not working

    #x_temp1, x_temp2, x_temp3, x_temp4 = X_train[y_train_new == 1], X_train[y_train_new == 2], X_train[y_train_new == 3], X_train[y_train_new == 4]
    x_temp1 = np.vstack([X_train[y_train_new==1]]*2)
    x = np.concatenate((X_train[y_train_new==0],x_temp1))
    y = np.concatenate((np.zeros(len(X_train[y_train_new==0])),np.ones(len(x_temp1))))
    x_temp1 = None
    x_temp2 = np.vstack([X_train[y_train_new == 2]] * 4)
    x = np.concatenate((x,x_temp2))
    y = np.concatenate((y,np.ones(len(x_temp2))*2))
    x_temp2=None

    x_temp3 = np.vstack([X_train[y_train_new == 3]] * 6)
    x = np.concatenate((x, x_temp3))
    y = np.concatenate((y, np.ones(len(x_temp3)) * 3))
    x_temp3 = None

    x_temp4 = np.vstack([X_train[y_train_new == 4]] * 6)
    x = np.concatenate((x, x_temp4))
    y = np.concatenate((y, np.ones(len(x_temp4)) * 4))
    x_temp4 = None

    """
    x_temp3 = np.vstack([X_train[y_train_new == 3]] * 6)
    x_temp4 = np.vstack([X_train[y_train_new == 4]] * 6)

    y_temp1 = np.ones(len(x_temp1))
    y_temp2 = np.ones(len(x_temp2))
    y_temp3 = np.ones(len(x_temp3))
    y_temp4 = np.ones(len(x_temp4))

    x = np.concatenate((X_train[y_train_new==0], x_temp1,x_temp2,x_temp3,x_temp4))
    del x_temp1, x_temp2,x_temp3,x_temp4

    y = np.concatenate((np.zeros(len(X_train[y_train_new==0])),y_temp1,y_temp2,y_temp3,y_temp4))
"""
    print("x_temp",np.shape(x),"y_temp=",np.shape(y))
    return x,y
    #y_temp1  = np.ones(len(x_temp1)) * 2,  np.ones(len(x_temp1)) * 4,  np.ones(len(x_temp1)) * 7,  np.ones(len(x_temp1)) * 7
    #X_train = np.concatenate((X_train, x_temp))
    #y_train_new = np.concatenate((y_train_new, y_temp))


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=0):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    print("begin preprocess")
    X_train = rgb2gray(X_train)
    X_train = X_train[:, np.newaxis, :, :]

    X_valid = rgb2gray(X_valid)
    X_valid = X_valid[:, np.newaxis, :, :]

    y_train_new = np.zeros(len(y_train))
    print(np.shape(y_train_new))
    for i in range(len(y_train)):
        y_train_new[i] = action_to_id(y_train[i])

    y_valid_new = np.zeros(len(y_valid))
    for i in range(len(y_valid)):
        y_valid_new[i] = action_to_id(y_valid[i])


    # print("after sampling, X-train",np.shape(X_train),"y_train=",np.shape(y_train_new))

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    if (history_length > 0):
        x_train_new = np.zeros((len(X_train), 1 + history_length, 96, 96))
        x_valid_new = np.zeros((len(X_valid), 1 + history_length, 96, 96))
        # print("x_train_new shape=",np.shape(x_train_new))
        for i, img in enumerate(X_train):
            temp = X_train[i]
            for n in range(history_length):
                if i > n:
                    n = n + 1
                    temp = np.concatenate((temp, X_train[i - n]), axis=0)
                else:
                    # for the intial images where sufficient history is not present,
                    # I attach the same scene as input to the network
                    # print("X_train shape", np.shape(X_train[i]),"n=", n)
                    temp = np.concatenate((temp, X_train[i]), axis=0)
            x_train_new[i] = temp

        for i, img in enumerate(X_valid):
            temp = X_valid[i]
            for n in range(history_length):
                if i > n:
                    n = n + 1
                    temp = np.concatenate((temp, X_valid[i - n]), axis=0)
                else:
                    temp = np.concatenate((temp, X_valid[i]), axis=0)
            x_valid_new[i] = temp
        #X_train, y_train_new = upsampling2(x_train_new, y_train_new)
        return x_train_new, y_train_new, x_valid_new, y_valid_new

    #X_train,y_train_new = upsampling2(X_train, y_train_new)

    print("X_train shape", np.shape(X_train), "y_train shape=", np.shape(y_train))
    print("end preprocess")

    return X_train, y_train_new, X_valid, y_valid_new


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models",
                tensorboard_dir="./tensorboard", history_length=0):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    # agent = BCAgent(...)
    agent = BCAgent(history_length,lr = lr)

    tensorboard_eval = Evaluation(tensorboard_dir, name="Training" + str(history_length) + "images",
                                  stats=["train_loss", "train_acc", "val_loss",
                                         "val_acc"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    print("X_train =", np.shape(X_train), "Y_train=", np.shape(y_train))
    print("X_valid =", np.shape(X_valid), "Y_valid=", np.shape(y_valid))

    def sample_minibatch(X, y, batch_size):
        # X_mini= X[i*batch_size : i*batch_size + batch_size]
        # y_mini = y[i*batch_size : i*batch_size + batch_size]

        idx = np.random.randint(0, X.shape[0], batch_size)
        X_mini = X[idx]
        y_mini = y[idx]
        return X_mini, y_mini

    jump_size = 10
    # creating arrays to store data to be used for matplotlib plotting later ..
    train_loss_data = np.zeros(int(n_minibatches / jump_size) )
    train_acc_data = np.zeros(int(n_minibatches / jump_size) )
    val_loss_data = np.zeros(int(n_minibatches / jump_size) )
    val_acc_data = np.zeros(int(n_minibatches / jump_size) )
    for i in range(n_minibatches):
        print("batch", i)

        X, y = sample_minibatch(X_train, y_train, batch_size)
        print("X_mini shape", np.shape(X), "y_mini", np.shape(y))
        loss, acc = agent.update(X, y)
        X_v, y_v = sample_minibatch(X_valid, y_valid, 300)
        val_loss, val_acc = agent.predict_val(X_v, y_v)
        print("train_loss=", loss, "train_acc=", acc, "val_loss=", val_loss, "val_acc=", val_acc)
        if i % jump_size == 0:
            # compute training/ validation accuracy and write it to tensorboard
            # tensorboard_eval.write_episode_data(...)
            print("Batch=", i, "Loss=", loss, "Acc=", acc)
            train_loss_data[int(i / jump_size)] = loss
            val_loss_data[int(i / jump_size)] = val_loss
            train_acc_data[int(i / jump_size)] = acc
            val_acc_data[int(i / jump_size)] = val_acc

            tensorboard_eval.write_episode_data(episode=i, eval_dict={"train_loss": loss, "train_acc": acc,
                                                                      "val_loss": val_loss, "val_acc": val_acc})

    # TODO: save your agent
    print("train_loss_data",train_loss_data)
    print("val_loss_data",val_loss_data)
    print("train_acc_data",train_acc_data)
    print("val_acc_data",val_acc_data)
    plt.plot(train_loss_data, label="Train Loss")
    plt.plot(val_loss_data, label="Val Loss")
    plt.xlabel("Minibatches (x10)")
    #plt.xticks(np.arange(0,801,10))
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_data, label="Train Acc.")
    plt.plot(val_acc_data, label="Val Acc")
    plt.xlabel("Minibatches (x10)")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    model_dir = agent.save(os.path.join(model_dir, "agent_0hist_800runs_wo_upsampl.pt"))
    print("Model saved in file: %s" % model_dir)


def upsampling(X_train, y_train):
    # causing the RAM to run out of memory..
    # find workaround

    x_t_0 = X_train[y_train == 0]
    x_t_1 = X_train[y_train == 1]
    x_t_1 = resample(x_t_1, replace=True, n_samples=len(x_t_1) * 3)
    x_t_2 = X_train[y_train == 2]
    x_t_2 = resample(x_t_2, replace=True, n_samples=len(x_t_2) * 4)
    x_t_3 = X_train[y_train == 3]
    x_t_3 = resample(x_t_3, replace=True, n_samples=len(x_t_3) * 7)
    x_t_4 = X_train[y_train == 4]
    x_t_4 = resample(x_t_4, replace=True, n_samples=len(x_t_4) * 7)

    y_t_0 = np.zeros(len(x_t_0))
    y_t_1 = np.zeros(len(x_t_1)) + 1
    y_t_2 = np.zeros(len(x_t_2)) + 2
    y_t_3 = np.zeros(len(x_t_3)) + 3
    y_t_4 = np.zeros(len(x_t_4)) + 4

    X = np.concatenate([x_t_0, x_t_1, x_t_2, x_t_3, x_t_4])
    y = np.concatenate([y_t_0, y_t_1, y_t_2, y_t_3, y_t_4])

    del x_t_0, x_t_1, x_t_2, x_t_3, x_t_4, y_t_0, y_t_1, y_t_2, y_t_3, y_t_4

    print("After sampling,Shape X=", np.shape(X), "y=", np.shape(y))

    return X, y


if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    history_length = 0
    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,
                                                       history_length=history_length)

    # X_train, y_train= upsampling(X_train, y_train)
    # #train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=800, batch_size=64, lr=0.001,history_length=history_length)
