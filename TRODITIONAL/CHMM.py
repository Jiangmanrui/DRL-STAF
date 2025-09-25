# chmm.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DATAPREPRO import prepro
from tqdm import tqdm


# ---------- 工具函数 ----------
def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def gaussian_logpdf(x, mu, var):
    eps = 1e-8
    var = np.maximum(var, eps)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


class CoupledHMM3:
    """
    3条时间序列的 Coupled Hidden Markov Model:
    - 每条链有 K 个隐状态，总联合状态数 S = K^3
    - 发射：GMM（每链每状态 C 个高斯分量；C=1 时退化为单高斯）
    - 转移：联合状态之间的全矩阵 A (S x S)，可选 net 投影结构
    """

    def __init__(self, K=3, num_nodes=3, max_iter=50, tol=1e-4, seed=0, net=None,
                 mix_components=1,base_prior=0.0):
        self.net = net
        self.parents = None
        self.K = K
        self.M = int(num_nodes)  # NEW: 链条数
        self.S = K ** self.M  # CHANGED: 用 M
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.base_prior = float(base_prior)

        self.pi = None
        self.A = None

        # 保留但不用于发射（统一用 M）
        self.mu = None  # (M,K)
        self.var = None  # (M,K)

        # GMM 参数（统一用 M）
        self.C = int(mix_components)
        self.mix_w = None  # (M, K, C)
        self.mix_mu = None  # (M, K, C)
        self.mix_var = None  # (M, K, C)

        self._build_index_maps()

    def _build_index_maps(self):
        K, M = self.K, self.M
        shape = (K,) * M
        # idx -> (z1,...,zM)
        self.idx2tuple = np.array(np.unravel_index(np.arange(self.S), shape)).T  # (S, M

    # ---------- 概率计算 ----------
    def _emission_loglik_t(self, y_t):
        S, K, C, M = self.S, self.K, self.C, self.M
        eps = 1e-8

        # (M,K): 每链每状态的 loglik（GMM 的 logsumexp）
        chain_state_loglik = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                lw = np.log(self.mix_w[m, k] + eps)  # (C,)
                var = np.maximum(self.mix_var[m, k], eps)  # (C,)
                lN = -0.5 * (np.log(2 * np.pi * var) + (y_t[m] - self.mix_mu[m, k]) ** 2 / var)
                chain_state_loglik[m, k] = logsumexp(lw + lN)

        ll = np.zeros(S)
        for s in range(S):
            z = self.idx2tuple[s]  # (M,)
            # sum over chains
            ll[s] = np.sum([chain_state_loglik[m, z[m]] for m in range(M)])
        return ll

    def _forward_backward(self, Y):
        T = Y.shape[0]
        S = self.S

        logB = np.zeros((T, S))
        for t in range(T):
            logB[t] = self._emission_loglik_t(Y[t])

        log_alpha = np.zeros((T, S))
        log_alpha[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, T):
            la = log_alpha[t - 1][:, None] + np.log(self.A + 1e-12)
            log_alpha[t] = logB[t] + logsumexp(la, axis=0)

        loglik = logsumexp(log_alpha[-1], axis=0)

        log_beta = np.zeros((T, S))
        for t in range(T - 2, -1, -1):
            lb = np.log(self.A + 1e-12) + (logB[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = logsumexp(lb, axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((T - 1, S, S))
        for t in range(T - 1):
            m = (log_alpha[t][:, None]
                 + np.log(self.A + 1e-12)
                 + (logB[t + 1] + log_beta[t + 1])[None, :])
            m = m - logsumexp(m)  # normalize in log
            xi[t] = np.exp(m)

        return gamma, xi, loglik

    # ---------- M 步：参数更新 ----------
    def _m_step(self, Y, gamma, xi):
        T = Y.shape[0]
        S, K, C = self.S, self.K, self.C
        eps = 1e-8

        # 初始分布
        self.pi = gamma[0] / (np.sum(gamma[0]) + 1e-12)

        # 转移矩阵（带轻微平滑）
        A_num = np.sum(xi, axis=0)  # (S,S)
        A_num += 1e-2
        if self.net is None:
            A_den = np.sum(A_num, axis=1, keepdims=True) + 1e-12
            self.A = A_num / A_den
        else:
            self._project_A_with_net(A_num)

        # ===== 发射（GMM）更新 =====
        # 1) 边缘权重
        w_mk_t = np.zeros((self.M, K, T))
        for m in range(self.M):
            for k in range(K):
                mask_s = (self.idx2tuple[:, m] == k)
                w_mk_t[m, k] = np.sum(gamma[:, mask_s], axis=1)

        # 2) 责任累积
        new_w = np.zeros((self.M, K, C))
        sum_y = np.zeros((self.M, K, C))
        sum_y2 = np.zeros((self.M, K, C))
        for m in range(self.M):
            y = Y[:, m]
            for k in range(K):
                lw = np.log(self.mix_w[m, k] + eps)
                var = np.maximum(self.mix_var[m, k], eps)
                lN = -0.5 * (np.log(2 * np.pi * var)[None, :] + ((y[:, None] - self.mix_mu[m, k][None, :]) ** 2) / var[
                                                                                                                   None,
                                                                                                                   :])
                log_post_unnorm = lw[None, :] + lN
                log_norm = logsumexp(log_post_unnorm, axis=1)[:, None]
                q_tc = np.exp(log_post_unnorm - log_norm)
                wt = w_mk_t[m, k][:, None]
                r_tc = wt * q_tc
                new_w[m, k] += np.sum(r_tc, axis=0)
                sum_y[m, k] += (r_tc * y[:, None]).sum(axis=0)
                sum_y2[m, k] += (r_tc * (y[:, None] ** 2)).sum(axis=0)
                new_w[m, k] = np.maximum(new_w[m, k], eps)

        # 3) 更新
        self.mix_w = new_w / (new_w.sum(axis=-1, keepdims=True) + eps)
        self.mix_mu = sum_y / new_w
        self.mix_var = np.maximum(sum_y2 / new_w - self.mix_mu ** 2, 1e-6)

    # ---------- 训练 ----------
    def fit(self, Y):
        # parents 列表
        if self.net is not None:
            self.parents = [list(np.where(self.net[:, m] == 1)[0]) for m in range(self.M)]
        else:
            self.parents = [[] for _ in range(self.M)]

        Y = np.asarray(Y, dtype=float)
        T = Y.shape[0]
        S, K, C, M = self.S, self.K, self.C, self.M

        self.pi = np.full(S, 1.0 / S)
        A = self.rng.random((S, S))
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
        self.A = 0.9 * A + 0.1 * (1.0 / S)

        # 占位（不用于发射）
        self.mu = np.zeros((M, K))
        self.var = np.zeros((M, K))

        # GMM 初始化 (M,K,C)
        self.mix_w = np.full((M, K, C), 1.0 / C)
        self.mix_mu = np.zeros((M, K, C))
        self.mix_var = np.zeros((M, K, C))
        for m in range(M):
            q = np.quantile(Y[:, m], np.linspace(0, 1, K + 2)[1:-1])
            centers = np.concatenate([[Y[:, m].min()], q, [Y[:, m].max()]])
            state_means = np.linspace(centers[0], centers[-1], K)
            for k in range(K):
                base = state_means[k]
                if C == 1:
                    self.mix_mu[m, k, 0] = base
                    self.mix_var[m, k, 0] = np.var(Y[:, m]) / K + 1e-2
                else:
                    offsets = np.linspace(-0.5, 0.5, C)
                    self.mix_mu[m, k, :] = base + 0.1 * np.std(Y[:, m]) * offsets
                    self.mix_var[m, k, :] = (np.var(Y[:, m]) / (K * C)) + 1e-2

        prev_ll = -np.inf
        for it in range(self.max_iter):
            gamma, xi, ll = self._forward_backward(Y)
            self._m_step(Y, gamma, xi)
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def _project_A_with_net(self, A_num):
        K, S, M = self.K, self.S, self.M
        counts = []
        for m in range(M):
            Pm = len(self.parents[m])
            shape = (K,) * Pm + (K,) + (K,)
            counts.append(np.zeros(shape))

        base_counts = [np.zeros((K, K)) for _ in range(M)]

        for s in range(S):
            z = self.idx2tuple[s]  # (M,)
            par_vals = [tuple(z[j] for j in self.parents[m]) for m in range(M)]
            row = A_num[s]
            if row.sum() == 0:
                continue
            for s2 in range(S):
                w = row[s2]
                if w <= 0:
                    continue
                z2 = self.idx2tuple[s2]
                for m in range(M):
                    key = par_vals[m] + (z[m], z2[m])
                    counts[m][key] += w
                    base_counts[m][z[m], z2[m]] += w

        # base_prior（若>0）作为“本地转移”伪计数
        if self.base_prior > 0.0:
            for m in range(M):
                row_sum = base_counts[m].sum(axis=1, keepdims=True) + 1e-12
                base_prior_prob = base_counts[m] / row_sum
                if counts[m].ndim == 2:
                    counts[m] += self.base_prior * base_prior_prob
                else:
                    for parent_idx in np.ndindex(counts[m].shape[:-2]):
                        counts[m][parent_idx] += self.base_prior * base_prior_prob

        thetas = [c / (c.sum(axis=-1, keepdims=True) + 1e-12) for c in counts]

        A_new = np.zeros((S, S))
        for s in range(S):
            z = self.idx2tuple[s]
            par_vals = [tuple(z[j] for j in self.parents[m]) for m in range(M)]
            for s2 in range(S):
                z2 = self.idx2tuple[s2]
                prod = 1.0
                for m in range(M):
                    sl = par_vals[m] + (z[m], z2[m])
                    prod *= thetas[m][sl]
                A_new[s, s2] = prod

        A_new = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-12)
        self.A = A_new

    # ---------- Viterbi 解码 ----------
    def viterbi(self, Y):
        Y = np.asarray(Y, dtype=float)
        T, S, M = Y.shape[0], self.S, self.M

        logB = np.zeros((T, S))
        for t in range(T):
            logB[t] = self._emission_loglik_t(Y[t])

        log_delta = np.zeros((T, S))
        psi = np.zeros((T, S), dtype=int)

        log_delta[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, T):
            Mx = log_delta[t - 1][:, None] + np.log(self.A + 1e-12)
            psi[t] = np.argmax(Mx, axis=0)
            log_delta[t] = np.max(Mx, axis=0) + logB[t]

        path_joint = np.zeros(T, dtype=int)
        path_joint[-1] = np.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            path_joint[t] = psi[t + 1, path_joint[t + 1]]

        z = self.idx2tuple[path_joint]  # (T,M)
        # 返回 list: 长度 M，每个元素 shape=(T,)
        return [z[:, m] for m in range(self.M)]

    # ---------- 预测 ----------
    def forecast(self, Y, n_steps=1):
        Y = np.asarray(Y, dtype=float)
        T, S, K, C, M = Y.shape[0], self.S, self.K, self.C, self.M

        logB_T = np.zeros((T, S))
        for t in range(T):
            logB_T[t] = self._emission_loglik_t(Y[t])

        log_alpha = np.zeros((T, S))
        log_alpha[0] = np.log(self.pi + 1e-12) + logB_T[0]
        for t in range(1, T):
            la = log_alpha[t - 1][:, None] + np.log(self.A + 1e-12)
            log_alpha[t] = logB_T[t] + logsumexp(la, axis=0)
        alpha_T = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))

        state_proba = np.zeros((n_steps, S))
        obs_mean = np.zeros((n_steps, M))
        obs_var = np.zeros((n_steps, M))
        map_state = np.zeros(n_steps, dtype=int)

        # 预计算混合后的均值/方差: (M,K)
        mix_mean = np.sum(self.mix_w * self.mix_mu, axis=-1)
        mix_var = np.sum(self.mix_w * (self.mix_var + (self.mix_mu - mix_mean[..., None]) ** 2), axis=-1)

        p = alpha_T.copy()
        for h in range(n_steps):
            p = p @ self.A
            p = p / (p.sum() + 1e-12)
            state_proba[h] = p
            map_state[h] = np.argmax(p)

            for m in range(M):
                pm = np.zeros(K)
                for k in range(K):
                    s_mask = (self.idx2tuple[:, m] == k)
                    pm[k] = p[s_mask].sum()
                pm = pm / (pm.sum() + 1e-12)

                mean_m = np.sum(pm * mix_mean[m])
                exp_var = np.sum(pm * mix_var[m])
                var_mean = np.sum(pm * (mix_mean[m] - mean_m) ** 2)
                obs_mean[h, m] = mean_m
                obs_var[h, m] = exp_var + var_mean

        map_tuple = self.idx2tuple[map_state]  # (n_steps,M)
        # 为了与之前一致，返回 map_state_chains 仍给三元组风格的解包：改为 list
        return {
            "state_proba": state_proba,
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "map_state_joint": map_state,
            "map_state_chains": [map_tuple[:, m] for m in range(M)],
        }


# ------------------ 示例：合成数据 + 训练 + 预测 ------------------

# ==== 配置 ====
# env_name = 'sim_chosmm_5000_g2_2_0.2'
# net = np.array([[0, 1, 0],
#                 [1, 0, 1],
#                 [0, 1, 0]])

# env_name = 'sim_chosmm_10_10000_g2_2_0.2'
# net = np.load("G:/mypro/predict_and_states/data/sim_chosmm_W_10.npy")

# env_name = 'sim_chosmm_50_10000_g2_2_0.2'
# net = np.load("G:/mypro/predict_and_states/data/sim_chosmm_W_50.npy")

# env_name = 'HL2'
env_name = 'machine'
net = None

directory = "./preTrained/{}".format(env_name)  # save trained models
directory2 = "./results/{}".format(env_name)  # save results
os.makedirs(directory, exist_ok=True)
os.makedirs(directory2, exist_ok=True)

filename = "CHMM_" + env_name

# ==== 读数据 ====
data, data_max, data_min, data_label = prepro(env_name, None)

T = data.shape[0]

# ==== 划分 ====
train_end = int(0.8 * T)
test_beg = train_end

# ==== 训练 ====
K = 2
num_nodes = data.shape[1]
model = CoupledHMM3(K=K, num_nodes=num_nodes, max_iter=120, tol=1e-5, seed=0, net=net, mix_components=3,base_prior=30)  # C=2 可调
model.fit(data[:train_end])

# 全序列 Viterbi
z_hat_all = model.viterbi(data)   # list，长度 = num_nodes

# ==== 评估与存储 ====
START_AT = 1000
colnames = []
for n in range(num_nodes):
    colnames += [f"state{n}", f"label{n}", f"pred{n}", f"target{n}", f"mae{n}", f"mse{n}"]
colnames += ["mae", "mse"]
results = pd.DataFrame(columns=colnames)

t = START_AT
with tqdm(total=data.shape[0] - t) as pbar:
    while t < data.shape[0]:
        results = results._append({}, ignore_index=True)

        # 固定窗口以提速（你原来的2000）
        fc = model.forecast(data[max(t - 2000, 0):t], n_steps=1)
        pred_means = fc["obs_mean"][0]

        for nn in range(num_nodes):
            pred = pred_means[nn] * (data_max[nn] - data_min[nn]) + data_min[nn]
            target = data[t, nn].reshape(-1) * (data_max[nn] - data_min[nn]) + data_min[nn]

            results.loc[len(results) - 1, f"state{nn}"] = z_hat_all[nn][t]
            results.loc[len(results) - 1, f"label{nn}"] = data_label[t][nn]
            results.loc[len(results) - 1, f"pred{nn}"] = pred
            results.loc[len(results) - 1, f"target{nn}"] = target
            results.loc[len(results) - 1, f"mae{nn}"] = np.abs(pred - target)
            results.loc[len(results) - 1, f"mse{nn}"] = (pred - target) ** 2

            if nn == 0:
                results.loc[len(results) - 1, "mae"] = np.abs(pred - target)
                results.loc[len(results) - 1, "mse"] = (pred - target) ** 2
            else:
                results.loc[len(results) - 1, "mae"] = results.loc[len(results) - 1, "mae"] + np.abs(
                    pred - target)
                results.loc[len(results) - 1, "mse"] = results.loc[len(results) - 1, "mse"] + (
                        pred - target) ** 2

        results.loc[len(results) - 1, "mae"] /= num_nodes
        results.loc[len(results) - 1, "mse"] /= num_nodes

        t += 1
        pbar.update(1)

print(results)
results.to_csv(directory2 + "/" + filename + "_best.csv", index=False)
mae = np.average(results["mae"])
print(mae)

max_cols = 5
num_rows = int(np.ceil(num_nodes / max_cols))
fig, axes = plt.subplots(num_rows * 2, max_cols, figsize=(4 * max_cols, 6 * num_rows), sharex=True)
axes = np.atleast_2d(axes)
fig.suptitle(filename + "_" + str(mae))
for n in range(num_nodes):
    col = n % max_cols
    row_group = n // max_cols
    row_obs = row_group * 2
    row_state = row_group * 2 + 1

    axes[row_obs, col].plot(results[[f"label{n}", f"state{n}"]])
    axes[row_state, col].plot(results[[f"target{n}", f"pred{n}"]])

for j in range(num_nodes, num_rows * max_cols):
    col = j % max_cols
    row_group = j // max_cols
    fig.delaxes(axes[row_group * 2, col])
    fig.delaxes(axes[row_group * 2 + 1, col])
plt.show()
