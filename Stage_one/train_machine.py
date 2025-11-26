import copy
import torch
from PPO import PPO
from PREDMmachine import PREDM
import os
import pandas as pd
from ENV import Env
from tqdm import tqdm
import numpy as np
from DATAPREPRO import prepro
import matplotlib.pyplot as plt
import random

random_seed = 0

def set_seed(seed):
    print(f"[INFO] Set all seeds to {seed}")
    # Python 内置
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU/GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# ===== Default hyperparameters / settings =====
future_weight = 0.5
score1bs = 4
score2bs = 2
con_actions_theta = 8
con_theta = 0.015
chajubs = 2
window_size = [2, 2, 2]
history_dim = 2
hidden_dim = 4
param = [8, 2]
start = 1000
max_timesteps = 2000

tau = 0.01
num_states = 2
nn = 2

env_name = 'machine'

directory = "./preTrained/{}".format(env_name)  # save trained models
directory2 = "./results/{}".format(env_name)  # save trained models
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(directory2):
    os.makedirs(directory2)



def train(no=str(random_seed)):
    filename = no+"_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(max_timesteps,
                                                                     nn,
                                                                     future_weight,
                                                                     num_states,
                                                                     score1bs,
                                                                     score2bs,
                                                                     con_actions_theta,
                                                                     con_theta,
                                                                     chajubs,
                                                                     window_size,
                                                                     history_dim,
                                                                     hidden_dim,
                                                                     param,
                                                                     tau,
                                                                     env_name)
    print("filename", filename)

    actor_lr = 1e-4
    critic_lr = 1e-4
    pred_lr = 1e-4

    max_episodes = 5000
    gamma = 0.99
    lmbda = 0.95
    loops = 10
    loops2 = 100
    pat = 5
    eps = 0.2

    log_interval = 10

    train_size = 0.8
    val_size = 0
    pred_dim = 1

    val_ep_reward = -999
    val_changecount = 0

    data, targetdata, data_max, data_min, data_label = prepro(env_name, train_size, nn)

    try:
        feature_num = data.shape[1]
    except:
        feature_num = 1
    env = Env(time_series=data,
              target_time_series=targetdata,
              window_size=window_size,
              train_size=train_size,
              val_size=val_size,
              hidden_dim=hidden_dim,
              history_dim=history_dim,
              num_states=num_states,
              feature_num=feature_num,
              max_timesteps=max_timesteps)
    env.chajubs = chajubs
    env.con_actions_theta = con_actions_theta
    env.future_weight = future_weight
    env.con_theta = con_theta
    env.score1bs = score1bs
    env.score2bs = score2bs
    env.qs = start

    state_dim = {
        'S1': env.observation_space['S1'].shape,
        'S11': env.observation_space['S11'].shape,
        'S3': env.observation_space['S3'].shape,
        'S4': env.observation_space['S4'].shape,
    }
    action_dim = num_states
    policy = PPO(state_dim, hidden_dim, history_dim, action_dim, actor_lr, critic_lr, lmbda,
                 loops, eps, gamma, device)
    prednet = PREDM(state_dim, pred_dim, hidden_dim, num_states, pred_lr, loops2, pat, tau, param, device)

    log_f = open(filename + "_log.txt", "w+")

    actor_loss_ave = []
    critic_loss_ave = []

    max_val_reward = -9999
    change_countshz = []
    counts = 0

    reward_ave = []
    moving_ave = []

    totalstep = 0

    for episode in range(1, max_episodes + 1):
        env.con_theta_episode_count = episode

        transition_dict = {'state_S1': [], 'state_S2': [], 'state_S3': [], 'state_S4': [],
                           'probs': [], 'actions': [],
                           'nstate_S1': [], 'nstate_S2': [], 'nstate_S3': [], 'nstate_S4': [],
                           'rewards': [], 'dones': [],
                           'pred_state_S1': [], 'pred_state_S2': [], 'pred_target': [],
                           'pred_choice': []}
        # reset environment and train
        state = env.reset(0, prednet)
        t = 0
        ep_reward = 0
        with tqdm(total=max_timesteps) as pbar:
            while t < max_timesteps:
                probs, action = policy.take_action(state)
                next_state, reward, done, pred_train = env.step(state, probs, action, prednet)
                transition_dict['state_S1'].append(state["S11"])
                transition_dict['state_S3'].append(state["S3"])
                transition_dict['state_S4'].append(state["S4"])
                transition_dict['probs'].append(probs)
                transition_dict['actions'].append(action)
                transition_dict['nstate_S1'].append(next_state["S11"])
                transition_dict['nstate_S3'].append(next_state["S3"])
                transition_dict['nstate_S4'].append(next_state["S4"])
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                transition_dict['pred_state_S1'].append(pred_train["S1"])
                transition_dict['pred_target'].append(pred_train["target"])
                transition_dict['pred_choice'].append(pred_train["choice"])
                state = next_state
                ep_reward += reward
                totalstep = totalstep + 1
                t = t + 1
                pbar.update(1)
                if done or env.current_step >= env.max_steps:
                    break

        prednet.updatebase(transition_dict)
        if episode < 5:
            prednet.update(transition_dict, env.choicehz)
            episode += 1
            continue
        else:
            prednet.use_target_pred = True
            prednet.update2(transition_dict, env.chazhihz, env.choicehz, env.errorhz)

        actor_loss, critic_loss = policy.update(transition_dict)

        actor_loss_ave.append(actor_loss)
        critic_loss_ave.append(critic_loss)
        reward_ave.append(ep_reward)
        log_f.write('{},{},{},{},{},{},{}\n'.format(episode, ep_reward, critic_loss, actor_loss, val_ep_reward,
                                                    env.changecount, val_changecount))
        change_countshz.append(env.changecount)
        if len(change_countshz) > 100:
            change_countshz.pop(0)
        log_f.flush()
        if episode % log_interval == 0:

            # reset environment and validate
            state = env.reset(1, prednet)

            t2 = 0
            val_ep_reward = 0
            while t2 < env.val_size:
                probs, action = policy.take_action(state, False)
                next_state, reward, done, pred_train = env.step(state, probs, action, prednet)
                state = next_state
                val_ep_reward += reward

                t2 = t2 + 1
                if done or env.current_step >= env.max_steps:
                    break
            val_changecount = env.changecount
            print(val_ep_reward)

            print(filename)
            ave_reward_ave = np.average(reward_ave)
            ave_critic_loss_ave = np.average(critic_loss_ave)
            ave_actor_loss_ave = np.average(actor_loss_ave)

            moving_ave.append(val_ep_reward)
            if len(moving_ave) > 20:
                moving_ave.pop(0)

            ave_moving_ave = np.average(moving_ave)
            print("Episode: {}\tAverage Reward: {}\tAverage Loss: {},{}\tVAL: {}".format(episode,
                                                                                         ave_reward_ave,
                                                                                         ave_critic_loss_ave,
                                                                                         ave_actor_loss_ave,
                                                                                         val_ep_reward))

            if episode > 1000:
                print(ave_moving_ave, max_val_reward)
                if np.average(change_countshz) <= max_timesteps / num_states * 0.6:
                    if ave_moving_ave > max_val_reward:
                        print("save best")
                        max_val_reward = ave_moving_ave
                        counts = 0
                        policy.save(directory, filename + "_best")
                        prednet.save(directory, filename + "_best")
                    else:
                        if len(moving_ave) == 20:
                            counts += 1
                else:
                    counts = 0
            reward_ave = []
            actor_loss_ave = []
            critic_loss_ave = []

            print("counts", counts, np.average(change_countshz))
            if (counts >= 50):
                print("Early stopping at episode {}".format(episode))
                policy.load(directory, filename + "_best")
                prednet.load(directory, filename + "_best")

                results = pd.DataFrame(columns=["state", "label", "pred", "predbase", "target", "mae", "mse"])
                state = env.reset(2, prednet)
                t3 = 0
                test_ep_reward = 0
                with tqdm(total=env.test_size) as pbar:
                    while t3 < env.test_size:
                        probs, action = policy.take_action(state, False)
                        next_state, reward, done, pred_train = env.step(state, probs, action, prednet)
                        state = next_state
                        test_ep_reward += reward

                        pred = env.pred * (data_max - data_min) + data_min
                        predbase = env.predbase * (data_max - data_min) + data_min
                        target = env.state_pred['target'][0][0] * (data_max - data_min) + data_min
                        results = results._append({}, ignore_index=True)
                        results.iloc[-1, 0] = action
                        results.iloc[-1, 1] = data_label[env.current_step - 1]
                        results.iloc[-1, 2] = pred
                        results.iloc[-1, 3] = predbase
                        results.iloc[-1, 4] = target
                        results.iloc[-1, 5] = np.abs(pred - target)
                        results.iloc[-1, 6] = (pred - target) ** 2

                        t3 = t3 + 1
                        pbar.update(1)
                        if done or env.current_step >= env.max_steps:
                            break
                print(results)
                results.to_csv(directory2 + "/" + filename + "_best.csv")
                mae = np.average(results.loc[(env.train_size - 1000):, "mae"])
                print(mae)
                fig, axes = plt.subplots(2, 1, figsize=(15, 5))
                fig.suptitle(filename + "\nbest_" + str(test_ep_reward) + "_" + str(mae))
                axes[0].plot(results[["label", "state"]])
                axes[1].plot(results[["target", "pred", "predbase"]])
                plt.show()
                break

        if episode >= 200 and episode % 200 == 0:
            policy.save(directory, filename + "_" + str(episode))
            prednet.save(directory, filename + "_" + str(episode))

            results = pd.DataFrame(columns=["state", "label", "pred", "predbase", "target", "mae", "mse"])
            # reset environment and test
            state = env.reset(2, prednet)
            t3 = 0
            test_ep_reward = 0
            with tqdm(total=env.test_size) as pbar:
                while t3 < env.test_size:
                    probs, action = policy.take_action(state, False)
                    next_state, reward, done, pred_train = env.step(state, probs, action, prednet)
                    state = next_state
                    test_ep_reward += reward

                    pred = env.pred * (data_max - data_min) + data_min
                    predbase = env.predbase * (data_max - data_min) + data_min
                    target = env.state_pred['target'][0][0] * (data_max - data_min) + data_min
                    results = results._append({}, ignore_index=True)
                    results.iloc[-1, 0] = action
                    results.iloc[-1, 1] = data_label[env.current_step - 1]
                    results.iloc[-1, 2] = pred
                    results.iloc[-1, 3] = predbase
                    results.iloc[-1, 4] = target
                    results.iloc[-1, 5] = np.abs(pred - target)
                    results.iloc[-1, 6] = (pred - target) ** 2

                    t3 = t3 + 1
                    pbar.update(1)
                    if done or env.current_step >= env.max_steps:
                        break
            print(results)
            results.to_csv(directory2 + "/" + filename + "_" + str(episode) + ".csv")
            mae = np.average(results.loc[(env.train_size - 1000):, "mae"])
            print(mae)

            fig, axes = plt.subplots(2, 1, figsize=(15, 5))
            fig.suptitle(filename + "\n" + str(episode) + "_" + str(test_ep_reward) + "_" + str(mae))
            axes[0].plot(results[["label", "state"]])
            axes[1].plot(results[["target", "pred", "predbase"]])
            plt.show()
    env.close()
    log_f.close()

if __name__ == '__main__':
    set_seed(random_seed)
    train()
