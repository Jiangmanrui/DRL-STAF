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


class IndependentHMMs:
    """
    Independent HMMs for M chains (no coupling):
      - Each chain m has K hidden states, GMM emissions with C components.
      - Parameters per chain: pi[m,K], A[m,K,K], mix_w/mu/var[m,K,C]
      - E/M steps and Viterbi run independently per chain.
    """
    def __init__(self, K=2, num_nodes=3, mix_components=3, max_iter=100, tol=1e-5, seed=0):
        self.K = int(K)
        self.M = int(num_nodes)
        self.C = int(mix_components)
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        # Per-chain params
        self.pi = [None]*self.M            # list of (K,)
        self.A  = [None]*self.M            # list of (K,K)
        self.mix_w  = [None]*self.M        # list of (K,C)
        self.mix_mu = [None]*self.M        # list of (K,C)
        self.mix_var= [None]*self.M        # list of (K,C)

    # ---------- emissions (GMM) ----------
    def _state_loglik_chain(self, y, m):
        """
        For chain m:
          y: (T,) observations
        returns:
          logB: (T, K) where logB[t,k] = log p(y_t | z_t=k)
        """
        K, C = self.K, self.C
        T = y.shape[0]
        logB = np.zeros((T, K))
        eps = 1e-8
        w  = self.mix_w[m]   # (K,C)
        mu = self.mix_mu[m]  # (K,C)
        va = np.maximum(self.mix_var[m], eps)  # (K,C)

        # log N(y|mu,var) for all (t,k,c)
        # shape (T, K, C)
        y2 = y[:, None, None]
        lN = -0.5*(np.log(2*np.pi*va)[None, :, :] + (y2 - mu[None, :, :])**2/va[None, :, :])
        logB = logsumexp(np.log(w[None, :, :]+eps) + lN, axis=2)  # sum over components
        return logB

    # ---------- forward-backward per chain ----------
    def _forward_backward_chain(self, y, m):
        """
        Returns gamma (T,K), xi (T-1,K,K), loglik for chain m
        """
        K = self.K
        T = y.shape[0]
        logB = self._state_loglik_chain(y, m)  # (T,K)

        logA = np.log(self.A[m] + 1e-12)       # (K,K)
        logpi= np.log(self.pi[m] + 1e-12)      # (K,)

        log_alpha = np.zeros((T, K))
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            la = log_alpha[t-1][:, None] + logA  # (K,K)
            log_alpha[t] = logB[t] + logsumexp(la, axis=0)

        loglik = logsumexp(log_alpha[-1])

        log_beta = np.zeros((T, K))
        for t in range(T-2, -1, -1):
            lb = logA + (logB[t+1] + log_beta[t+1])[None, :]
            log_beta[t] = logsumexp(lb, axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            mtx = (log_alpha[t][:, None] + logA + (logB[t+1]+log_beta[t+1])[None, :])
            mtx -= logsumexp(mtx)
            xi[t] = np.exp(mtx)

        return gamma, xi, loglik

    # ---------- M step per chain ----------
    def _m_step_chain(self, y, gamma, xi, m):
        K, C = self.K, self.C
        T = y.shape[0]
        eps = 1e-8

        # pi, A
        self.pi[m] = gamma[0] / (np.sum(gamma[0]) + 1e-12)
        A_num = np.sum(xi, axis=0) + 1e-6
        A_den = np.sum(A_num, axis=1, keepdims=True) + 1e-12
        self.A[m] = A_num / A_den

        # GMM updates
        # weights per state-time: w_tk = gamma[t,k]
        # E-step for mixture within each state k
        new_w  = np.zeros((K, C))
        sum_y  = np.zeros((K, C))
        sum_y2 = np.zeros((K, C))

        # current params
        w  = self.mix_w[m]    # (K,C)
        mu = self.mix_mu[m]   # (K,C)
        va = self.mix_var[m]  # (K,C)
        va = np.maximum(va, eps)

        for k in range(K):
            # responsibilities q_tc within state k
            # shape (T,C)
            lN = -0.5*(np.log(2*np.pi*va[k])[None, :] + ((y[:, None]-mu[k][None, :])**2)/va[k][None, :])
            log_post = np.log(w[k]+eps)[None, :] + lN
            log_norm = logsumexp(log_post, axis=1)[:, None]
            q_tc = np.exp(log_post - log_norm)     # (T,C)

            r_tc = gamma[:, k][:, None] * q_tc     # (T,C)
            new_w[k]  = np.sum(r_tc, axis=0)
            sum_y[k]  = np.sum(r_tc * y[:, None], axis=0)
            sum_y2[k] = np.sum(r_tc * (y[:, None]**2), axis=0)

        new_w = np.maximum(new_w, eps)
        self.mix_w[m]   = new_w / (new_w.sum(axis=1, keepdims=True) + eps)
        self.mix_mu[m]  = sum_y / new_w
        self.mix_var[m] = np.maximum(sum_y2 / new_w - self.mix_mu[m]**2, 1e-6)

    # ---------- fit all chains ----------
    def fit(self, Y):
        """
        Y: (T, M) numpy array
        """
        Y = np.asarray(Y, dtype=float)
        T, M, K, C = Y.shape[0], self.M, self.K, self.C

        # init per-chain params
        for m in range(M):
            self.pi[m] = np.full(K, 1.0/K)
            A = self.rng.random((K, K))
            A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
            self.A[m] = 0.9*A + 0.1*(1.0/K)

            # GMM init: K*C components per chain
            self.mix_w[m]  = np.full((K, C), 1.0/C)
            self.mix_mu[m] = np.zeros((K, C))
            self.mix_var[m]= np.zeros((K, C))
            y = Y[:, m]
            q = np.quantile(y, np.linspace(0, 1, K+2)[1:-1])
            centers = np.concatenate([[y.min()], q, [y.max()]])
            state_means = np.linspace(centers[0], centers[-1], K)
            for k in range(K):
                base = state_means[k]
                if C == 1:
                    self.mix_mu[m][k, 0]  = base
                    self.mix_var[m][k, 0] = np.var(y)/K + 1e-2
                else:
                    offsets = np.linspace(-0.5, 0.5, C)
                    self.mix_mu[m][k, :]  = base + 0.1*np.std(y)*offsets
                    self.mix_var[m][k, :] = (np.var(y)/(K*C)) + 1e-2

        prev_ll = -np.inf
        for it in range(self.max_iter):
            ll_total = 0.0
            gammas, xis = [], []
            # E-step per chain
            for m in range(M):
                g, x, ll = self._forward_backward_chain(Y[:, m], m)
                gammas.append(g)
                xis.append(x)
                ll_total += ll
            # M-step per chain
            for m in range(M):
                self._m_step_chain(Y[:, m], gammas[m], xis[m], m)

            if np.abs(ll_total - prev_ll) < self.tol:
                break
            prev_ll = ll_total
        return self

    # ---------- viterbi per chain ----------
    def viterbi(self, Y):
        Y = np.asarray(Y, dtype=float)
        T, M, K = Y.shape[0], self.M, self.K
        paths = []
        for m in range(M):
            y = Y[:, m]
            logB = self._state_loglik_chain(y, m)       # (T,K)
            logA = np.log(self.A[m] + 1e-12)            # (K,K)
            logpi= np.log(self.pi[m] + 1e-12)           # (K,)

            delta = np.zeros((T, K))
            psi = np.zeros((T, K), dtype=int)

            delta[0] = logpi + logB[0]
            for t in range(1, T):
                Mx = delta[t-1][:, None] + logA
                psi[t] = np.argmax(Mx, axis=0)
                delta[t] = np.max(Mx, axis=0) + logB[t]

            path = np.zeros(T, dtype=int)
            path[-1] = np.argmax(delta[-1])
            for t in range(T-2, -1, -1):
                path[t] = psi[t+1, path[t+1]]
            paths.append(path)
        return paths  # list of length M, each (T,)

    # ---------- forecast n_steps per chain ----------
    def forecast(self, Y_hist, n_steps=1):
        """
        Y_hist: (T_hist, M)  —— 仅用于得到每条链的滤波分布 alpha_T
        返回:
          state_proba: None (独立链不返回联合分布)
          obs_mean:    (n_steps, M)
          obs_var:     (n_steps, M)
          map_state_chains: list长度M，每条链MAP状态（每步）
        """
        Y_hist = np.asarray(Y_hist, dtype=float)
        T, M, K, C = Y_hist.shape[0], self.M, self.K, self.C

        obs_mean = np.zeros((n_steps, M))
        obs_var  = np.zeros((n_steps, M))
        map_state_chains = [np.zeros(n_steps, dtype=int) for _ in range(M)]

        # 组件混合后的 state-level 发射均值/方差
        mix_mean = [np.sum(self.mix_w[m]*self.mix_mu[m], axis=1) for m in range(M)]  # (K,)
        mix_var  = [np.sum(self.mix_w[m]*(self.mix_var[m] + (self.mix_mu[m]-mix_mean[m][:,None])**2), axis=1)
                    for m in range(M)]  # (K,)

        for m in range(M):
            # filter to get alpha_T on chain m
            y = Y_hist[:, m]
            logB = self._state_loglik_chain(y, m)    # (T,K)
            logA = np.log(self.A[m] + 1e-12)
            logpi= np.log(self.pi[m] + 1e-12)

            log_alpha = np.zeros((T, K))
            log_alpha[0] = logpi + logB[0]
            for t in range(1, T):
                la = log_alpha[t-1][:, None] + logA
                log_alpha[t] = logB[t] + logsumexp(la, axis=0)
            alpha_T = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))  # (K,)

            p = alpha_T.copy()
            for h in range(n_steps):
                p = p @ self.A[m]                 # next-step state dist
                p = p / (p.sum() + 1e-12)
                map_state_chains[m][h] = np.argmax(p)

                # mixture over states -> obs mean/var
                mean_m = np.sum(p * mix_mean[m])
                exp_var = np.sum(p * mix_var[m])
                var_mean = np.sum(p * (mix_mean[m] - mean_m)**2)
                obs_mean[h, m] = mean_m
                obs_var[h, m]  = exp_var + var_mean

        return {
            "state_proba": None,
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "map_state_chains": map_state_chains,
        }



# ------------------ 示例：合成数据 + 训练 + 预测 ------------------

# ==== 配置 ====
# env_name = 'sim_chosmm_5000_g2_2_0.2'
# net = np.array([[0, 1, 0],
#                 [1, 0, 1],
#                 [0, 1, 0]])
#
# env_name = 'sim_chosmm_10_10000_g2_2_0.2'
# net = np.load("G:/mypro/predict_and_states/data/sim_chosmm_W_10.npy")

# env_name = 'sim_chosmm_50_10000_g2_2_0.2'
# net = np.load("G:/mypro/predict_and_states/data/sim_chosmm_W_50.npy")

# env_name = 'HL2'
env_name = 'machine'

directory = "./preTrained/{}".format(env_name)  # save trained models
directory2 = "./results/{}".format(env_name)  # save results
os.makedirs(directory, exist_ok=True)
os.makedirs(directory2, exist_ok=True)

filename = "HMM_" + env_name

# ==== 读数据 ====
data, data_max, data_min, data_label = prepro(env_name, None)

T = data.shape[0]

# ==== 划分 ====
train_end = int(0.8 * T)
test_beg = train_end

# ==== 训练 ====
K = 2
num_nodes = data.shape[1]
model = IndependentHMMs(K=K, num_nodes=num_nodes, mix_components=3, max_iter=120, tol=1e-5, seed=0)
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
