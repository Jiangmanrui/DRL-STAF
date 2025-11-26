import gym
from gym import spaces
import numpy as np
import copy
import math

# similar to stage_one

def inverse_softmax(arr, window, T=1.0):
    c = np.cumsum(np.r_[0, arr])
    res = c[window:] - c[:-window]
    x = -res / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inverse_softmax2(arr, T=1.0):
    x = -arr / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Env(gym.Env):

    def __init__(self, time_series, target_time_series, window_size, train_size, val_size, hidden_dim, history_dim,
                 num_states, feature_num,
                 max_timesteps):
        super(Env, self).__init__()

        self.time_series = time_series
        self.target_time_series = target_time_series
        self.max_steps = len(time_series)
        self.max_timesteps = max_timesteps
        self.window_size = window_size
        self.train_size = int(self.max_steps * train_size)
        self.val_size = self.train_size

        self.qs = np.max(self.window_size) + 1

        self.hidden_dim = hidden_dim
        self.history_dim = history_dim
        self.num_states = num_states
        self.history = {}
        self.actionhz = []
        self.probhz = []
        self.errorhz = []
        self.choicehz = []
        self.last_action = -9999
        self.con_actions = 1
        self.con_actions_theta = 8
        self.con_theta = 0.02
        self.con_theta_episode_count = 0
        self.con_theta_episode_count_yz = 4000
        self.con_theta_max = 0.05
        self.chajubs = 1
        self.score1bs = 1
        self.score2bs = 1
        self.future_weight = 0.5
        self.feature_num = feature_num
        self.com_p1 = np.ones((self.train_size - np.max(self.window_size) - 2)) * (-99.0)
        self.com_p2 = np.ones((self.train_size - np.max(self.window_size) - 2)) * (-9999.0)
        self.com_p3 = np.ones((self.train_size - self.max_timesteps - np.max(self.window_size) - 1)) * (-99.0)
        self.sort_p = inverse_softmax2(self.com_p3)
        self.reward_hz = 0
        self.msetongjihz = 0

        self.action_space = spaces.Discrete(self.num_states)

        self.observation_space = spaces.Dict({
            'S1': spaces.Box(low=-1, high=1, shape=(window_size[0], self.feature_num), dtype=np.float32),
            'S11': spaces.Box(low=-1, high=1, shape=(window_size[1], self.feature_num), dtype=np.float32),
            'S2': spaces.Box(low=-1, high=1, shape=(self.num_states,), dtype=np.float32),
            'S3': spaces.Box(low=-1, high=1, shape=(self.num_states,), dtype=np.float32),
            'S4': spaces.Box(low=-1, high=1, shape=(self.num_states,), dtype=np.float32),
        })

        self.current_step = window_size

    def reset(self, type, prednet, current_step):
        self.actionhz = []
        self.probhz = []
        self.errorhz = []
        self.choicehz = []
        self.changecount = 0
        self.reward_hz = 0
        self.current_step = current_step
        self.current_start = current_step
        self.type = type
        if type == 0:
            self.max_steps = self.current_step + self.max_timesteps
            self.choice = np.zeros((1, self.num_states), dtype=np.float32)
            self.choice[0][np.random.choice(range(self.num_states))] = 1
            self.con_actions = 1
        elif type == 1:
            self.max_steps = self.train_size
        else:
            self.current_step = self.qs
            # print(self.current_step)
            self.max_steps = len(self.time_series)
            # print(self.max_steps)
            self.test_size = self.max_steps - self.qs
            self.choice = np.zeros((1, self.num_states), dtype=np.float32)
            self.choice[0][np.random.choice(range(self.num_states))] = 1
        self.last_action = -9999
        self.s3 = np.ones((self.history_dim, self.num_states), dtype=np.float32) / self.num_states
        self.s4 = np.zeros((self.history_dim, self.num_states), dtype=np.float32)

        self.state_pred = self._get_observation_pred()
        pred, pred_k = prednet.predict(self.state_pred, self.choice)
        predbase = prednet.predictbase(self.state_pred)
        self.errorbase = (predbase - self.state_pred['target']) ** 2
        error_k = (pred_k - self.state_pred['target']) ** 2
        # print(error_k)
        self.s4 = np.vstack((self.s4, error_k.reshape((1, self.num_states))))
        self.s4 = np.delete(self.s4, 0, axis=0)
        return self._get_observation()

    def pred_yxl_data(self):
        current_step = np.max(self.window_size) + 1
        s1hz = []
        targethz = []
        while current_step + 1 < self.train_size:
            s1 = self.time_series[current_step - self.window_size[0] - 1:current_step - 1].reshape(
                (-1, self.feature_num))
            target = self.target_time_series[current_step - 1].reshape((1, -1))
            s1hz.append(s1)
            targethz.append(target)
            current_step += 1
        return s1hz, targethz

    def _get_observation_pred(self):
        s1 = self.time_series[self.current_step - self.window_size[0] - 1:self.current_step - 1].reshape(
            (-1, self.feature_num))
        target = self.target_time_series[self.current_step - 1].reshape((1, -1))
        return {'S1': s1, 'target': target}

    def _get_observation(self):
        s1 = self.time_series[self.current_step - self.window_size[0]:self.current_step].reshape((-1, self.feature_num))
        s11 = self.time_series[self.current_step - self.window_size[1]:self.current_step].reshape(
            (-1, self.feature_num))
        return {'S1': s1, 'S11': s11, 'S3': self.s3, 'S4': self.s4}

    def step(self, state, probs, action, prednet, rewardbase=0, pred_sr=None):
        reward2 = 0
        if self.last_action < 0:
            self.last_action = action
        else:
            if action == self.last_action:
                self.con_actions += 1
            else:
                self.changecount += 1
                self.last_action = action
                if self.con_actions < self.con_actions_theta:
                    reward2 = -self.con_theta * (self.con_actions_theta - self.con_actions) / (
                            self.con_actions_theta - 1)
                    # reward2 = -0.1
                self.con_actions = 1

        actiong = np.zeros((1, self.num_states), dtype=np.float32)
        actiong[0][action] = 1
        self.actionhz.append(action)
        self.probhz.append(probs)
        self.errorhz.append(self.s4[-1, :])

        pred_train = copy.deepcopy(self.state_pred)
        pred_train['choice'] = copy.deepcopy(actiong[0])
        self.choice = actiong
        self.choicehz.append(actiong)
        if pred_sr==None:
            pred, pred_k = prednet.predict(state, actiong)
            predbase = prednet.predictbase(state)
        else:
            pred_k = pred_sr[0]
            pred = pred_k[action]
            # print(pred_k.shape)
            predbase = pred_sr[1]
        self.pred = pred
        self.pred_k = pred_k
        self.predbase = predbase
        target = self.target_time_series[self.current_step].reshape((1, -1))

        self.state_pred = {k: state[k] for k in ['S1']}
        self.state_pred['target'] = target
        error_k = (pred_k - target) ** 2
        errorbase = (predbase - target) ** 2
        reward = self.score1bs * (
                self.future_weight * (errorbase[0, 0] - error_k[0, action]) + (1 - self.future_weight) * (
                self.errorbase[0, 0] - self.s4[-1, action])) + reward2 - rewardbase
        msetongji = -(errorbase[0, 0] + self.errorbase[0, 0])
        self.errorbase = errorbase

        self.s3 = np.vstack((self.s3, probs.reshape((1, self.num_states))))
        self.s3 = np.delete(self.s3, 0, axis=0)

        self.s4 = np.vstack((self.s4, error_k.reshape((1, self.num_states))))
        self.s4 = np.delete(self.s4, 0, axis=0)

        self.reward_hz += reward
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if self.type == 0:
            self.com_p1[self.current_step - 1 - np.max(self.window_size) - 1] = reward
            self.com_p2[self.current_step - 1 - np.max(self.window_size) - 1] = msetongji

        if done:
            self.actionhz = np.array(self.actionhz)
            self.probhz = np.concatenate(self.probhz)
            self.errorhz = np.array(self.errorhz)
            self.choicehz = np.array(self.choicehz)

            is_one_states = np.argmax(self.probhz, axis=1)

            self.errorbc = copy.deepcopy(self.errorhz)
            for k in range(self.num_states):
                errormin = copy.deepcopy(self.errorhz)
                errormin = np.delete(errormin, k, axis=1)
                errormin = np.min(errormin, axis=1)
                self.errorbc[:, k] = errormin - self.errorbc[:, k]
            errorls = copy.deepcopy(self.errorhz)
            errorls[list(range(len(errorls))), self.actionhz] = 999
            self.chazhihz = np.min(errorls, axis=1) - self.errorhz[list(range(len(errorls))), self.actionhz]

            def safe_average(arr, default=0.0):
                return np.average(arr) if arr.size > 0 else default

            if len(np.unique(is_one_states)) == 1:
                print("prefer one state")
            if len(np.unique(self.actionhz)) > 1:
                tongji = np.array([len(self.actionhz[self.actionhz == a]) for a in np.unique(self.actionhz)]) / len(
                    self.actionhz)
                print(tongji)

                s_errors_mean = []
                us_errors_mean = []
                for s in range(self.num_states):
                    mask1 = (self.actionhz == s)
                    mask2 = (self.actionhz != s)
                    indices1 = np.where(mask1)[0]
                    indices2 = np.where(mask2)[0]
                    errors_in_x = self.errorhz[indices1, s]
                    errors_notin_x = self.errorhz[indices2, s]

                    s_errors_mean.append(safe_average(errors_in_x))
                    us_errors_mean.append(safe_average(errors_notin_x))

                print("mse_selected_mean", s_errors_mean)
                print("mse_unselected_mean", us_errors_mean)
                chaju = np.array(us_errors_mean) - np.array(s_errors_mean)

                state_ave_chazhi = np.sum(np.abs(np.array(s_errors_mean)[:, np.newaxis] - np.array(s_errors_mean))) / 2
                chaju[chaju < 0] *= self.chajubs
                reward_final = -np.average(s_errors_mean) + chaju.sum() - state_ave_chazhi * self.score2bs
            else:
                reward_final = - 100
                print("all actions are the same")
            reward = reward + reward_final
            if self.type == 0:
                self.com_p3[self.current_start - np.max(self.window_size) - 1] = reward_final
                self.sort_p = (0.5 * (0.2 * inverse_softmax(self.com_p1, self.max_timesteps)
                                      + 0.8 * inverse_softmax(self.com_p2, self.max_timesteps)
                                      ) +
                               0.5 * inverse_softmax2(self.com_p3))
                self.sort_p = self.sort_p / np.sum(self.sort_p)
                self.com_p1[self.com_p1 > 0] = self.com_p1[self.com_p1 > 0] * 0.9999
                self.com_p1[self.com_p1 < 0] = self.com_p1[self.com_p1 < 0] * 1.0001
                self.com_p2 = self.com_p2 * 1.0001
                self.com_p3[self.com_p3 > 0] = self.com_p3[self.com_p3 > 0] * 0.9999
                self.com_p3[self.com_p3 < 0] = self.com_p3[self.com_p3 < 0] * 1.0001
        next_state = self._get_observation()

        return next_state, reward, done, pred_train
