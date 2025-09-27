import copy

random_seed = 0
import torch

torch.manual_seed(0)
from PPO_2 import PPO as PPO_2
from PPO_1 import PPO as PPO_1
from PREDMexchange import PREDM
import os
import pandas as pd
from ENV_base import Env as Env_base
from ENV import Env
from tqdm import tqdm
import numpy as np
from DATAPREPRO import prepro
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# HL
future_weight = 0.5
score1bs = 4
score2bs = 2
con_actions_theta = 8
con_theta = 0.015
con_theta_base = 0.015
chajubs = 2
window_size = [4,4,4]
history_dim = 2
hidden_dim = 4
param = [8, 2]
start = 1000
max_timesteps = 2000

tau = 0.01
num_states = 2
nn = 0

env_name = 'exchange'
net = np.ones((3, 3)) - np.eye(3)
directory = "./preTrained/{}".format(env_name)  # save trained models
directory2 = "./results/{}".format(env_name)  # save trained models
directory_base = "../Stage_one/PPOG_fs/preTrained/{}".format(env_name)  # save trained models
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(directory2):
    os.makedirs(directory2)
filename = "{}_netnone_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(max_timesteps,
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


def train():
    actor_lr = 1e-4
    critic_lr = 1e-4
    pred_lr = 1e-4
    max_episodes = 10000
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
    val_ep_reward_base = -999

    data, targetdata, data_max, data_min, data_label = prepro(env_name, train_size)
    num_nodes = data.shape[1]
    try:
        feature_num = data.shape[2]
    except:
        feature_num = 1

    filename_base = {}
    env_base = {}
    policy_base = {}
    prednet = {}
    state_dim_base = {}
    for nn in range(num_nodes):
        filename_base[nn] = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(2000,
                                                                                  nn,
                                                                                  future_weight,
                                                                                  num_states,
                                                                                  score1bs,
                                                                                  score2bs,
                                                                                  con_actions_theta,
                                                                                  con_theta_base,
                                                                                  chajubs,
                                                                                  window_size,
                                                                                  history_dim,
                                                                                  hidden_dim,
                                                                                  param,
                                                                                  tau,
                                                                                  env_name)
        env_base[nn] = Env_base(time_series=data[:, nn],
                                target_time_series=targetdata[:, nn],
                                window_size=window_size,
                                train_size=train_size,
                                val_size=val_size,
                                hidden_dim=hidden_dim,
                                history_dim=history_dim,
                                num_states=num_states,
                                feature_num=feature_num,
                                max_timesteps=max_timesteps)
        env_base[nn].chajubs = chajubs
        env_base[nn].con_actions_theta = con_actions_theta
        env_base[nn].future_weight = future_weight
        env_base[nn].con_theta = con_theta
        env_base[nn].score1bs = score1bs
        env_base[nn].score2bs = score2bs
        env_base[nn].qs = start

        state_dim_base[nn] = {
            'S1': env_base[nn].observation_space['S1'].shape,  # 获取S1的维度
            'S11': env_base[nn].observation_space['S11'].shape,  # 获取S1的维度
            'S3': env_base[nn].observation_space['S3'].shape,  # 获取S1的维度
            'S4': env_base[nn].observation_space['S4'].shape,  # 获取S1的维度
        }

        policy_base[nn] = PPO_1(state_dim_base[nn], hidden_dim, history_dim, num_states, actor_lr, critic_lr, lmbda,
                                loops, eps, gamma, device)
        prednet[nn] = PREDM(state_dim_base[nn], pred_dim, hidden_dim, num_states, pred_lr, loops2, pat, tau, param,
                            device)

        policy_base[nn].load(directory_base, filename_base[nn] + "_best")
        prednet[nn].load(directory_base, filename_base[nn] + "_best")

    env = Env(time_series=data,
              target_time_series=targetdata,
              window_size=window_size,
              train_size=train_size,
              val_size=val_size,
              hidden_dim=hidden_dim,
              history_dim=history_dim,
              num_states=num_states,
              num_nodes=num_nodes,
              feature_num=feature_num,
              max_timesteps=max_timesteps)
    env.chajubs = chajubs
    env.con_actions_theta = con_actions_theta
    env.future_weight = future_weight
    env.con_theta = con_theta
    env.score1bs = score1bs
    env.score2bs = score2bs

    env_real = copy.deepcopy(env_base)

    state_dim = {
        'S1': env.observation_space['S1'].shape,  # 获取S1的维度
        'S3': env.observation_space['S3'].shape,
        'S4': env.observation_space['S4'].shape,
        'S5': env.observation_space['S5'].shape,
    }
    action_dim = num_states
    policy = PPO_2(state_dim, hidden_dim, history_dim, action_dim, actor_lr, critic_lr, lmbda,
                   loops, eps, gamma, net, device)

    log_f = open(filename + "_log.txt", "w+")

    actor_loss_ave = []
    critic_loss_ave = []
    reward_ave = []
    moving_ave = []
    reward_ave2 = []
    max_val_reward = -9999
    change_countshz = np.zeros((100, num_nodes))
    counts = 0
    totalstep = 0
    train_sig = np.zeros(num_nodes)
    for episode in range(1, max_episodes + 1):
        transition_dict = {'state_S1': [], 'state_S2': [], 'state_S3': [], 'state_S4': [],
                           'state_S5': [],
                           'probs': [], 'actions': [],
                           'nstate_S1': [], 'nstate_S2': [], 'nstate_S3': [], 'nstate_S4': [],
                           'nstate_S5': [],
                           'rewards': [], 'dones': []}
        transition_dict2 = {}
        env.reset(0)
        state_base = {}
        state_real = {}
        for nn in range(num_nodes):
            env_base[nn].con_theta_episode_count = episode
            env_real[nn].con_theta_episode_count = episode
            state_base[nn] = env_base[nn].reset(0, prednet[nn], env.current_step)
            state_real[nn] = env_real[nn].reset(0, prednet[nn], env.current_step)
            transition_dict2[nn] = {'pred_state_S1': [], 'pred_state_S2': [], 'pred_target': [],
                                    'pred_choice': []}
        t = 0
        ep_reward = np.zeros(num_nodes)
        ep_reward_base = np.zeros(num_nodes)
        with tqdm(total=max_timesteps) as pbar:
            while t < max_timesteps:
                state_S1 = []
                state_S2 = []
                state_S3 = []
                state_S4 = []
                state_S5 = []
                nstate_S2 = []
                nstate_S5 = []
                sort_p = []
                reward_base_hz = []
                for nn in range(num_nodes):
                    probs_base, action_base = policy_base[nn].take_action(state_base[nn], False)
                    state_S2.append(env_base[nn].s3)
                    next_state_base, reward_base, done, pred_train = env_base[nn].step(
                        state_base[nn],
                        probs_base,
                        action_base,
                        prednet[nn])
                    # print("probs_base_hz.shape",probs_base_hz.shape)
                    state_S1.append(state_real[nn]["S11"])
                    state_S3.append(state_real[nn]["S3"])
                    state_S4.append(state_real[nn]["S4"])
                    state_S5.append(probs_base)
                    reward_base_hz.append(reward_base)

                    next_probs_base, next_action_base = policy_base[nn].take_action(next_state_base, False)

                    nstate_S2.append(env_base[nn].s3)
                    nstate_S5.append(next_probs_base)

                    state_base[nn] = next_state_base
                    ep_reward_base[nn] += reward_base

                state_S1 = np.stack(state_S1, axis=0)
                state_S2 = np.stack(state_S2, axis=0)
                state_S3 = np.stack(state_S3, axis=0)
                state_S4 = np.stack(state_S4, axis=0)
                state_S5 = np.stack(state_S5, axis=0)

                state = {
                    "S1": state_S1,
                    "S2": state_S2,
                    "S3": state_S3,
                    "S4": state_S4,
                    "S5": state_S5
                }

                probs, action = policy.take_action(state)
                nstate_S1 = []
                nstate_S3 = []
                nstate_S4 = []
                reward_hz = []
                changecounts = []
                for nn in range(num_nodes):
                    # print("probs.shape", probs[nn].shape)
                    next_state, reward, done, pred_train = env_real[nn].step(
                        state_real[nn],
                        probs[nn],
                        action[nn],
                        prednet[nn],
                        reward_base_hz[nn],
                        pred_sr=[env_base[nn].pred_k, env_base[nn].predbase])
                    if done:
                        changecounts.append(env_real[nn].changecount)
                        change_countshz[episode % 100][nn] = env_real[nn].changecount
                    nstate_S1.append(next_state["S11"])
                    nstate_S3.append(next_state["S3"])
                    nstate_S4.append(next_state["S4"])
                    reward_hz.append(reward)

                    sort_p.append(env_real[nn].sort_p)

                    transition_dict2[nn]['pred_state_S1'].append(pred_train["S1"])
                    transition_dict2[nn]['pred_target'].append(pred_train["target"])
                    transition_dict2[nn]['pred_choice'].append(pred_train["choice"])

                    state_real[nn] = next_state
                    ep_reward[nn] += reward

                nstate_S1 = np.stack(nstate_S1, axis=0)
                nstate_S2 = np.stack(nstate_S2, axis=0)
                nstate_S3 = np.stack(nstate_S3, axis=0)
                nstate_S4 = np.stack(nstate_S4, axis=0)
                nstate_S5 = np.stack(nstate_S5, axis=0)
                reward_hz = np.stack(reward_hz, axis=0)
                next_state = {
                    "S1": nstate_S1,
                    "S2": nstate_S2,
                    "S3": nstate_S3,
                    "S4": nstate_S4,
                    "S5": nstate_S5,
                }
                env.step(sort_p)

                transition_dict['state_S1'].append(state["S1"])
                transition_dict['state_S2'].append(state["S2"])
                transition_dict['state_S3'].append(state["S3"])
                transition_dict['state_S4'].append(state["S4"])
                transition_dict['state_S5'].append(state["S5"])
                transition_dict['probs'].append(probs)
                transition_dict['actions'].append(action)
                transition_dict['nstate_S1'].append(next_state["S1"])
                transition_dict['nstate_S2'].append(next_state["S2"])
                transition_dict['nstate_S3'].append(next_state["S3"])
                transition_dict['nstate_S4'].append(next_state["S4"])
                transition_dict['nstate_S5'].append(next_state["S5"])
                transition_dict['rewards'].append(reward_hz)
                transition_dict['dones'].append(done)

                totalstep = totalstep + 1
                t = t + 1
                pbar.update(1)
                if done or env.current_step >= env.max_steps:
                    break

        if episode > 2000:
            for nn in range(num_nodes):
                if train_sig[nn] != 0:
                    prednet[nn].updatebase(transition_dict2[nn])
                    prednet[nn].update2(transition_dict2[nn], env_real[nn].chazhihz, env_real[nn].choicehz)
                else:
                    if ep_reward[nn] > 0:
                        train_sig[nn] += 1
                        prednet[nn].updatebase(transition_dict2[nn])
                        prednet[nn].update2(transition_dict2[nn], env_real[nn].chazhihz, env_real[nn].choicehz)

        if episode % 1000 == 0:
            policy.entropy_coef = np.max([policy.entropy_coef - 0.005, 0.01])
        actor_loss, critic_loss = policy.update(transition_dict)
        actor_loss_ave.append(actor_loss)
        critic_loss_ave.append(critic_loss)
        reward_ave.append(ep_reward)
        reward_ave2.append(ep_reward_base)
        log_f.write('{},{},{},{},{},{},{},{}\n'.format(episode, ep_reward_base, ep_reward, critic_loss, actor_loss,
                                                       val_ep_reward_base, val_ep_reward, changecounts))
        log_f.flush()
        if episode % log_interval == 0:
            env.reset(1)
            state_base = {}
            state_real = {}
            for nn in range(num_nodes):
                state_base[nn] = env_base[nn].reset(1, prednet[nn], env.current_step)
                state_real[nn] = env_real[nn].reset(1, prednet[nn], env.current_step)

            t2 = 0
            val_ep_reward = np.zeros(num_nodes)
            val_ep_reward_base = np.zeros(num_nodes)
            while t2 < env.val_size:
                state_S1 = []
                state_S2 = []
                state_S3 = []
                state_S4 = []
                state_S5 = []
                nstate_S2 = []
                nstate_S5 = []
                reward_base_hz = []
                for nn in range(num_nodes):
                    probs_base, action_base = policy_base[nn].take_action(state_base[nn], False)
                    state_S2.append(env_base[nn].s3)
                    next_state_base, reward_base, done, pred_train = env_base[nn].step(
                        state_base[nn],
                        probs_base,
                        action_base,
                        prednet[nn])
                    state_S1.append(state_real[nn]["S11"])
                    state_S3.append(state_real[nn]["S3"])
                    state_S4.append(state_real[nn]["S4"])
                    state_S5.append(probs_base)
                    reward_base_hz.append(reward_base)

                    next_probs_base, next_action_base = policy_base[nn].take_action(next_state_base, False)

                    nstate_S2.append(env_base[nn].s3)
                    nstate_S5.append(next_probs_base)

                    state_base[nn] = next_state_base
                    val_ep_reward_base[nn] += reward_base

                state_S1 = np.stack(state_S1, axis=0)
                state_S2 = np.stack(state_S2, axis=0)
                state_S3 = np.stack(state_S3, axis=0)
                state_S4 = np.stack(state_S4, axis=0)
                state_S5 = np.stack(state_S5, axis=0)
                state = {
                    "S1": state_S1,
                    "S2": state_S2,
                    "S3": state_S3,
                    "S4": state_S4,
                    "S5": state_S5
                }

                probs, action = policy.take_action(state, False)
                nstate_S1 = []
                nstate_S3 = []
                nstate_S4 = []
                for nn in range(num_nodes):
                    next_state, reward, done, pred_train = env_real[nn].step(
                        state_real[nn],
                        probs[nn],
                        action[nn],
                        prednet[nn],
                        reward_base_hz[nn],
                        pred_sr=[env_base[nn].pred_k, env_base[nn].predbase])

                    nstate_S1.append(next_state["S11"])
                    nstate_S3.append(next_state["S3"])
                    nstate_S4.append(next_state["S4"])

                    sort_p.append(env_real[nn].sort_p)

                    transition_dict2[nn]['pred_state_S1'].append(pred_train["S1"])
                    transition_dict2[nn]['pred_target'].append(pred_train["target"])
                    transition_dict2[nn]['pred_choice'].append(pred_train["choice"])

                    state_real[nn] = next_state
                    val_ep_reward[nn] += reward

                env.step(sort_p)

                t2 = t2 + 1
                if done or env.current_step >= env.max_steps:
                    break
            print(filename)

            moving_ave.append(val_ep_reward)
            if len(moving_ave) > 20:
                moving_ave.pop(0)

            ave_moving_ave = np.average(moving_ave)
            print("Episode: {}\tAverage Reward: {}\tAverage Loss: {},{}\tVAL: {}".format(episode,
                                                                                         [np.average(reward_ave),
                                                                                          np.average(reward_ave2)],
                                                                                         np.average(
                                                                                             critic_loss_ave),
                                                                                         np.average(actor_loss_ave),
                                                                                         [val_ep_reward,
                                                                                          val_ep_reward_base]))

            if episode > 2000:
                print(ave_moving_ave, max_val_reward)
                if np.all(np.average(change_countshz, axis=0) <= max_timesteps / num_states * 0.6):
                    if ave_moving_ave > max_val_reward:
                        print("save best")
                        max_val_reward = ave_moving_ave
                        counts = 0
                        policy.save(directory, filename + "_best")
                        for nn in range(num_nodes):
                            prednet[nn].save(directory, filename + "_" + str(nn) + "_best")
                    else:
                        if len(moving_ave) == 20:
                            counts += 1
                else:
                    counts = 0

            reward_ave = []
            reward_ave2 = []
            actor_loss_ave = []
            critic_loss_ave = []

            print("counts", counts, np.average(change_countshz, axis=0))
            if counts >= 50:
                print("Early stopping at episode {}".format(episode))
                policy.load(directory, filename + "_best")
                for nn in range(num_nodes):
                    prednet[nn].load(directory, filename + "_" + str(nn) + "_best")
                colnames = []
                for n in range(num_nodes):
                    colnames.append("state" + str(n))
                    colnames.append("label" + str(n))
                    colnames.append("pred" + str(n))
                    colnames.append("predbase" + str(n))
                    colnames.append("target" + str(n))
                    colnames.append("mae" + str(n))
                    colnames.append("mse" + str(n))
                colnames.append("mae")
                colnames.append("mse")
                results = pd.DataFrame(columns=colnames)
                env.reset(2)
                state_base = {}
                state_real = {}
                for nn in range(num_nodes):
                    state_base[nn] = env_base[nn].reset(2, prednet[nn], env.current_step)
                    state_real[nn] = env_real[nn].reset(2, prednet[nn], env.current_step)

                t3 = 0
                test_ep_reward = np.zeros(num_nodes)
                test_ep_reward_base = np.zeros(num_nodes)
                with tqdm(total=env.test_size) as pbar:
                    while t3 < env.test_size:
                        state_S1 = []
                        state_S2 = []
                        state_S3 = []
                        state_S4 = []
                        state_S5 = []
                        nstate_S2 = []
                        nstate_S5 = []
                        reward_base_hz = []
                        action_base_hz = []
                        for nn in range(num_nodes):
                            probs_base, action_base = policy_base[nn].take_action(state_base[nn], False)
                            state_S2.append(env_base[nn].s3)
                            next_state_base, reward_base, done, pred_train = env_base[nn].step(
                                state_base[nn],
                                probs_base,
                                action_base,
                                prednet[nn])
                            state_S1.append(state_real[nn]["S11"])
                            state_S3.append(state_real[nn]["S3"])
                            state_S4.append(state_real[nn]["S4"])
                            state_S5.append(probs_base)
                            reward_base_hz.append(reward_base)
                            action_base_hz.append(action_base)

                            next_probs_base, next_action_base = policy_base[nn].take_action(next_state_base, False)

                            nstate_S2.append(env_base[nn].s3)
                            nstate_S5.append(next_probs_base)

                            state_base[nn] = next_state_base
                            test_ep_reward_base[nn] += reward_base

                        state_S1 = np.stack(state_S1, axis=0)
                        state_S2 = np.stack(state_S2, axis=0)
                        state_S3 = np.stack(state_S3, axis=0)
                        state_S4 = np.stack(state_S4, axis=0)
                        state_S5 = np.stack(state_S5, axis=0)
                        state = {
                            "S1": state_S1,
                            "S2": state_S2,
                            "S3": state_S3,
                            "S4": state_S4,
                            "S5": state_S5
                        }

                        probs, action = policy.take_action(state, False)
                        nstate_S1 = []
                        nstate_S3 = []
                        nstate_S4 = []
                        results = results._append({}, ignore_index=True)
                        for nn in range(num_nodes):
                            next_state, reward, done, pred_train = env_real[nn].step(
                                state_real[nn],
                                probs[nn],
                                action[nn],
                                prednet[nn],
                                reward_base_hz[nn],
                                pred_sr=[env_base[nn].pred_k, env_base[nn].predbase])

                            nstate_S1.append(next_state["S11"])
                            nstate_S3.append(next_state["S3"])
                            nstate_S4.append(next_state["S4"])

                            sort_p.append(env_real[nn].sort_p)

                            transition_dict2[nn]['pred_state_S1'].append(pred_train["S1"])
                            transition_dict2[nn]['pred_target'].append(pred_train["target"])
                            transition_dict2[nn]['pred_choice'].append(pred_train["choice"])

                            state_real[nn] = next_state
                            test_ep_reward[nn] += reward

                            pred = env_real[nn].pred.reshape(-1) * (data_max[nn] - data_min[nn]) + data_min[nn]
                            predbase = env_real[nn].predbase.reshape(-1) * (data_max[nn] - data_min[nn]) + data_min[nn]
                            target = env_real[nn].state_pred['target'].reshape(-1) * (data_max[nn] - data_min[nn]) + \
                                     data_min[nn]

                            results.loc[len(results) - 1, "state" + str(nn)] = action[nn]
                            results.loc[len(results) - 1, "statebase" + str(nn)] = action_base_hz[nn]
                            results.loc[len(results) - 1, "label" + str(nn)] = \
                                data_label[env_real[nn].current_step - 1][nn]
                            results.loc[len(results) - 1, "pred" + str(nn)] = pred
                            results.loc[len(results) - 1, "predbase" + str(nn)] = predbase
                            results.loc[len(results) - 1, "target" + str(nn)] = target
                            results.loc[len(results) - 1, "mae" + str(nn)] = np.abs(pred - target)
                            results.loc[len(results) - 1, "mse" + str(nn)] = (pred - target) ** 2
                            if nn == 0:
                                results.loc[len(results) - 1, "mae"] = np.abs(pred - target)
                                results.loc[len(results) - 1, "mse"] = (pred - target) ** 2
                            else:
                                results.loc[len(results) - 1, "mae"] = results.loc[len(results) - 1, "mae"] + np.abs(
                                    pred - target)
                                results.loc[len(results) - 1, "mse"] = results.loc[len(results) - 1, "mse"] + (
                                        pred - target) ** 2
                        results.loc[len(results) - 1, "mae"] = results.loc[len(results) - 1, "mae"] / num_nodes
                        results.loc[len(results) - 1, "mse"] = results.loc[len(results) - 1, "mse"] / num_nodes

                        env.step(sort_p)
                        t3 = t3 + 1
                        pbar.update(1)
                        if done or env.current_step >= env.max_steps:
                            break

                print(results)
                results.to_csv(directory2 + "/" + filename + "_best.csv")
                mae = np.average(results["mae"])
                print(mae)
                max_cols = 5
                num_rows = int(np.ceil(num_nodes / max_cols))
                fig, axes = plt.subplots(num_rows * 2, max_cols, figsize=(4 * max_cols, 6 * num_rows), sharex=True)
                axes = np.atleast_2d(axes)
                fig.suptitle(
                    filename + "\n" + str(episode) + "_" + str(test_ep_reward_base) + "_" + str(
                        test_ep_reward) + "_" + str(
                        mae))
                for n in range(num_nodes):
                    col = n % max_cols  # 当前节点放在哪一列
                    row_group = n // max_cols  # 当前节点属于第几组
                    row_obs = row_group * 2  # 观测在偶数行
                    row_state = row_group * 2 + 1  # 状态在奇数行

                    axes[row_obs, col].plot(results[["label" + str(n), "statebase" + str(n), "state" + str(n)]])
                    axes[row_state, col].plot(results[["target" + str(n), "predbase" + str(n), "pred" + str(n)]])

                for j in range(num_nodes, num_rows * max_cols):
                    col = j % max_cols
                    row_group = j // max_cols
                    fig.delaxes(axes[row_group * 2, col])
                    fig.delaxes(axes[row_group * 2 + 1, col])
                plt.show()
                break

        if episode >= 200 and episode % 200 == 0:
            policy.save(directory, filename + "_" + str(episode))
            colnames = []
            for n in range(num_nodes):
                prednet[n].save(directory, filename + "_" + str(n) + "_" + str(episode))
                colnames.append("state" + str(n))
                colnames.append("label" + str(n))
                colnames.append("pred" + str(n))
                colnames.append("predbase" + str(n))
                colnames.append("target" + str(n))
                colnames.append("mae" + str(n))
                colnames.append("mse" + str(n))
            colnames.append("mae")
            colnames.append("mse")
            results = pd.DataFrame(columns=colnames)
            env.reset(2)
            state_base = {}
            state_real = {}
            for nn in range(num_nodes):
                state_base[nn] = env_base[nn].reset(2, prednet[nn], env.current_step)
                state_real[nn] = env_real[nn].reset(2, prednet[nn], env.current_step)

            t3 = 0
            test_ep_reward = np.zeros(num_nodes)
            test_ep_reward_base = np.zeros(num_nodes)
            with tqdm(total=env.test_size) as pbar:
                while t3 < env.test_size:
                    state_S1 = []
                    state_S2 = []
                    state_S3 = []
                    state_S4 = []
                    state_S5 = []
                    nstate_S2 = []
                    nstate_S5 = []
                    reward_base_hz = []
                    action_base_hz = []
                    for nn in range(num_nodes):
                        probs_base, action_base = policy_base[nn].take_action(state_base[nn], False)
                        state_S2.append(env_base[nn].s3)
                        next_state_base, reward_base, done, pred_train = env_base[nn].step(
                            state_base[nn],
                            probs_base,
                            action_base,
                            prednet[nn])
                        state_S1.append(state_real[nn]["S11"])
                        state_S3.append(state_real[nn]["S3"])
                        state_S4.append(state_real[nn]["S4"])
                        state_S5.append(probs_base)
                        reward_base_hz.append(reward_base)
                        action_base_hz.append(action_base)

                        next_probs_base, next_action_base = policy_base[nn].take_action(next_state_base, False)

                        nstate_S2.append(env_base[nn].s3)
                        nstate_S5.append(next_probs_base)

                        state_base[nn] = next_state_base
                        test_ep_reward_base[nn] += reward_base

                    state_S1 = np.stack(state_S1, axis=0)
                    state_S2 = np.stack(state_S2, axis=0)
                    state_S3 = np.stack(state_S3, axis=0)
                    state_S4 = np.stack(state_S4, axis=0)
                    state_S5 = np.stack(state_S5, axis=0)
                    state = {
                        "S1": state_S1,
                        "S2": state_S2,
                        "S3": state_S3,
                        "S4": state_S4,
                        "S5": state_S5
                    }

                    probs, action = policy.take_action(state, False)
                    nstate_S1 = []
                    nstate_S3 = []
                    nstate_S4 = []
                    results = results._append({}, ignore_index=True)
                    for nn in range(num_nodes):
                        next_state, reward, done, pred_train = env_real[nn].step(
                            state_real[nn],
                            probs[nn],
                            action[nn],
                            prednet[nn],
                            reward_base_hz[nn],
                            pred_sr=[env_base[nn].pred_k, env_base[nn].predbase])

                        nstate_S1.append(next_state["S11"])
                        nstate_S3.append(next_state["S3"])
                        nstate_S4.append(next_state["S4"])

                        sort_p.append(env_real[nn].sort_p)

                        transition_dict2[nn]['pred_state_S1'].append(pred_train["S1"])
                        transition_dict2[nn]['pred_target'].append(pred_train["target"])
                        transition_dict2[nn]['pred_choice'].append(pred_train["choice"])

                        state_real[nn] = next_state
                        test_ep_reward[nn] += reward

                        pred = env_real[nn].pred.reshape(-1) * (data_max[nn] - data_min[nn]) + data_min[nn]
                        predbase = env_real[nn].predbase.reshape(-1) * (data_max[nn] - data_min[nn]) + data_min[nn]
                        target = env_real[nn].state_pred['target'].reshape(-1) * (data_max[nn] - data_min[nn]) + \
                                 data_min[nn]

                        results.loc[len(results) - 1, "state" + str(nn)] = action[nn]
                        results.loc[len(results) - 1, "statebase" + str(nn)] = action_base_hz[nn]
                        results.loc[len(results) - 1, "label" + str(nn)] = \
                            data_label[env_real[nn].current_step - 1][nn]
                        results.loc[len(results) - 1, "pred" + str(nn)] = pred
                        results.loc[len(results) - 1, "predbase" + str(nn)] = predbase
                        results.loc[len(results) - 1, "target" + str(nn)] = target
                        results.loc[len(results) - 1, "mae" + str(nn)] = np.abs(pred - target)
                        results.loc[len(results) - 1, "mse" + str(nn)] = (pred - target) ** 2
                        if nn == 0:
                            results.loc[len(results) - 1, "mae"] = np.abs(pred - target)
                            results.loc[len(results) - 1, "mse"] = (pred - target) ** 2
                        else:
                            results.loc[len(results) - 1, "mae"] = results.loc[len(results) - 1, "mae"] + np.abs(
                                pred - target)
                            results.loc[len(results) - 1, "mse"] = results.loc[len(results) - 1, "mse"] + (
                                    pred - target) ** 2
                    results.loc[len(results) - 1, "mae"] = results.loc[len(results) - 1, "mae"] / num_nodes
                    results.loc[len(results) - 1, "mse"] = results.loc[len(results) - 1, "mse"] / num_nodes

                    env.step(sort_p)
                    t3 = t3 + 1
                    pbar.update(1)
                    if done or env.current_step >= env.max_steps:
                        break
            print(results)
            results.to_csv(directory2 + "/" + filename + "_" + str(episode) + ".csv")
            # mae = np.average(results.loc[env.val_size - env.val_size2:, "mae"])
            mae = np.average(results["mae"])
            print(mae)
            # print(results[["pred","target"]])
            max_cols = 5
            num_rows = int(np.ceil(num_nodes / max_cols))
            fig, axes = plt.subplots(num_rows * 2, max_cols, figsize=(4 * max_cols, 6 * num_rows), sharex=True)
            axes = np.atleast_2d(axes)
            fig.suptitle(
                filename + "\n" + str(episode) + "_" + str(test_ep_reward_base) + "_" + str(
                    test_ep_reward) + "_" + str(
                    mae))
            for n in range(num_nodes):
                col = n % max_cols  # 当前节点放在哪一列
                row_group = n // max_cols  # 当前节点属于第几组
                row_obs = row_group * 2  # 观测在偶数行
                row_state = row_group * 2 + 1  # 状态在奇数行

                axes[row_obs, col].plot(results[["label" + str(n), "statebase" + str(n), "state" + str(n)]])
                axes[row_state, col].plot(results[["target" + str(n), "predbase" + str(n), "pred" + str(n)]])

            for j in range(num_nodes, num_rows * max_cols):
                col = j % max_cols
                row_group = j // max_cols
                fig.delaxes(axes[row_group * 2, col])
                fig.delaxes(axes[row_group * 2 + 1, col])
            plt.show()
    env.close()
    log_f.close()


if __name__ == '__main__':
    train()
