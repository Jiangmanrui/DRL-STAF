# HOHMM.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from DATAPREPRO import prepro

# ---------- utils ----------
def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

# =============== 单条链：高阶 HMM（GMM 发射） ===============
class SingleHOHMM:
    def __init__(self, K=2, order=2, mix_components=3, max_iter=80, tol=1e-5, seed=0):
        """
        K: 状态数
        order: 高阶阶数 r（转移 P(z_t | z_{t-1},...,z_{t-r})）
        mix_components: 每状态 GMM 分量数
        """
        self.K = int(K)
        self.R = int(order)
        self.C = int(mix_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)

        # 扩展状态空间（上下文）：c_t = (z_{t-R+1},...,z_t) ∈ {0..K-1}^R
        self.S = self.K ** self.R
        self.idx2ctx = np.array(np.unravel_index(np.arange(self.S), (self.K,) * self.R)).T  # (S, R)
        # ctx -> idx
        self.ctx2idx = {tuple(row.tolist()): i for i, row in enumerate(self.idx2ctx)}

        # 为转移构造 next_ctx 索引：给定扩展状态 u（上下文）和新状态 k，下一扩展状态 v 的索引
        self.next_ctx = np.zeros((self.S, self.K), dtype=int)  # (S, K)
        for u in range(self.S):
            ctx = self.idx2ctx[u]
            for k in range(self.K):
                nxt = tuple(list(ctx[1:]) + [k])
                self.next_ctx[u, k] = self.ctx2idx[nxt]

        # 每个扩展状态的“当前真实状态”（上下文最后一位）
        self.last_of_ctx = self.idx2ctx[:, -1]  # (S,)

        # 参数：初始分布、扩展转移（由高阶张量 psi 生成）、GMM 发射
        self.pi = None           # (S,)
        self.psi = None          # 高阶转移张量，shape = (K,)*R + (K,)   parents->dest
        self.A = None            # 扩展转移矩阵 (S,S)，由 psi 计算得到
        self.mix_w = None        # (K, C)
        self.mix_mu = None       # (K, C)
        self.mix_var = None      # (K, C)

        # 预先为“按真实状态取扩展状态”的掩码
        self.mask_last = np.zeros((self.K, self.S), dtype=bool)
        for k in range(self.K):
            self.mask_last[k, self.last_of_ctx == k] = True

    # ---------- 发射对数似然（GMM） ----------
    def _emission_loglik_t(self, y_t):
        # y_t: scalar (单链)
        eps = 1e-8
        # 对每个真实状态 k 计算 GMM 对数似然，然后映射到扩展状态上
        loglik_k = np.zeros(self.K)
        for k in range(self.K):
            lw = np.log(self.mix_w[k] + eps)         # (C,)
            var = np.maximum(self.mix_var[k], eps)   # (C,)
            lN = -0.5 * (np.log(2*np.pi*var) + (y_t - self.mix_mu[k])**2 / var)  # (C,)
            loglik_k[k] = logsumexp(lw + lN)
        # 扩展到 (S,)
        return loglik_k[self.last_of_ctx]

    def _build_A_from_psi(self):
        """由 psi（parents->dest 的条件分布）构造扩展转移矩阵 A"""
        S, K = self.S, self.K
        A = np.zeros((S, S))
        for u in range(S):
            parents = tuple(self.idx2ctx[u].tolist())  # (R,)
            for k in range(K):
                v = self.next_ctx[u, k]
                A[u, v] = self.psi[parents + (k,)]
        # 行归一化（数值稳健）
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
        self.A = A

    # ---------- 前向后向（在扩展状态上） ----------
    def _forward_backward(self, y):
        """
        y: (T,) 单链观测
        我们从 t0 = R-1 开始做 E 步（前 R-1 个时刻不参与 EM）
        """
        T = len(y)
        if T < self.R:
            # 不足以形成一个上下文；退化使用单步
            t0 = 0
        else:
            t0 = self.R - 1
        Teff = T - t0

        logB = np.zeros((Teff, self.S))
        for t in range(Teff):
            logB[t] = self._emission_loglik_t(y[t + t0])

        log_alpha = np.zeros((Teff, self.S))
        log_alpha[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, Teff):
            la = log_alpha[t-1][:, None] + np.log(self.A + 1e-12)
            log_alpha[t] = logB[t] + logsumexp(la, axis=0)

        loglik = logsumexp(log_alpha[-1], axis=0)

        log_beta = np.zeros((Teff, self.S))
        for t in range(Teff-2, -1, -1):
            lb = np.log(self.A + 1e-12) + (logB[t+1] + log_beta[t+1])[None, :]
            log_beta[t] = logsumexp(lb, axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)  # (Teff, S)

        xi = np.zeros((Teff-1, self.S, self.S))
        for t in range(Teff-1):
            m = (log_alpha[t][:, None]
                 + np.log(self.A + 1e-12)
                 + (logB[t+1] + log_beta[t+1])[None, :])
            m -= logsumexp(m)
            xi[t] = np.exp(m)

        return t0, gamma, xi, loglik

    # ---------- M-step ----------
    def _m_step(self, y, t0, gamma, xi):
        eps = 1e-8
        Teff = gamma.shape[0]

        # pi（扩展初始分布）
        self.pi = gamma[0] / (gamma[0].sum() + 1e-12)

        # 高阶转移 psi（parents->dest）
        # 统计 counts[parents, k] = sum_t sum_{u in parents} sum_{v with last=k and shift-consistent} xi[t,u,v]
        shape = (self.K,) * self.R + (self.K,)
        counts = np.zeros(shape)
        for t in range(Teff - 1):
            Xi_t = xi[t]  # (S,S)
            # 只考虑“合法迁移”：v 必须等于 next_ctx[u, k]
            # 否则 A[u,v]=0，Xi_t[u,v] 也会接近 0
            for u in range(self.S):
                parents = tuple(self.idx2ctx[u].tolist())
                for k in range(self.K):
                    v = self.next_ctx[u, k]
                    w = Xi_t[u, v]
                    if w > 0:
                        counts[parents + (k,)] += w

        # 归一化得到 psi
        denom = counts.sum(axis=-1, keepdims=True) + 1e-12
        self.psi = counts / denom

        # 用 psi 重建 A
        self._build_A_from_psi()

        # ===== 发射（GMM）更新 =====
        # 对每个真实状态 k：权重 w_tk = sum_{contexts last=k} gamma[t, context]
        T = len(y)
        # 聚合到真实时间轴 t = t0..T-1
        w_tk = np.zeros((T, self.K))
        w_tk[t0:] = gamma @ self.mask_last.T  # (Teff,S) @ (S,K) -> (Teff,K)

        # 统计充分量（每个状态 k，GMM EM 的一次迭代）
        new_w = np.zeros((self.K, self.C))
        sum_y = np.zeros((self.K, self.C))
        sum_y2 = np.zeros((self.K, self.C))

        for k in range(self.K):
            # r_tc = w_tk[:,k] * q_tc
            lw = np.log(self.mix_w[k] + eps)            # (C,)
            var = np.maximum(self.mix_var[k], eps)      # (C,)
            lN = -0.5*(np.log(2*np.pi*var)[None, :] + ((y[:, None]-self.mix_mu[k][None, :])**2)/var[None, :])  # (T,C)
            log_post = lw[None, :] + lN
            log_Z = logsumexp(log_post, axis=1)[:, None]
            q_tc = np.exp(log_post - log_Z)             # (T,C)

            wt = w_tk[:, k][:, None]                    # (T,1)
            r_tc = wt * q_tc                            # (T,C)

            new_w[k]  = r_tc.sum(axis=0)
            sum_y[k]  = (r_tc * y[:, None]).sum(axis=0)
            sum_y2[k] = (r_tc * (y[:, None]**2)).sum(axis=0)

            new_w[k] = np.maximum(new_w[k], eps)

        self.mix_w  = new_w / (new_w.sum(axis=-1, keepdims=True) + eps)
        self.mix_mu = sum_y / new_w
        self.mix_var= np.maximum(sum_y2 / new_w - self.mix_mu**2, 1e-6)

    # ---------- 训练 ----------
    def fit(self, y):
        """
        y: (T,) 当前链的观测
        """
        y = np.asarray(y, dtype=float)
        T = len(y)

        # 初始化 GMM
        self.mix_w  = np.full((self.K, self.C), 1.0/self.C)
        self.mix_mu = np.zeros((self.K, self.C))
        self.mix_var= np.zeros((self.K, self.C))
        q = np.quantile(y, np.linspace(0, 1, self.K+2)[1:-1])
        centers = np.concatenate([[y.min()], q, [y.max()]])
        state_means = np.linspace(centers[0], centers[-1], self.K)
        for k in range(self.K):
            base = state_means[k]
            if self.C == 1:
                self.mix_mu[k, 0]  = base
                self.mix_var[k, 0] = np.var(y)/self.K + 1e-2
            else:
                offsets = np.linspace(-0.5, 0.5, self.C)
                self.mix_mu[k, :]  = base + 0.1*np.std(y) * offsets
                self.mix_var[k, :] = np.var(y)/(self.K*self.C) + 1e-2

        # 初始化 psi（均匀），由此构造 A；pi 均匀
        self.psi = np.full((self.K,)*self.R + (self.K,), 1.0/self.K)
        self._build_A_from_psi()
        self.pi = np.full(self.S, 1.0/self.S)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            t0, gamma, xi, ll = self._forward_backward(y)
            self._m_step(y, t0, gamma, xi)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    # ---------- Viterbi ----------
    def viterbi(self, y):
        y = np.asarray(y, dtype=float)
        T = len(y)
        if T < self.R:
            # 退化：直接用 pi 的 MAP 上下文，重复最后一位
            z = np.zeros(T, dtype=int)
            return z

        t0 = self.R - 1
        Teff = T - t0

        logB = np.zeros((Teff, self.S))
        for t in range(Teff):
            logB[t] = self._emission_loglik_t(y[t + t0])

        log_delta = np.zeros((Teff, self.S))
        psi_ptr = np.zeros((Teff, self.S), dtype=int)

        log_delta[0] = np.log(self.pi + 1e-12) + logB[0]
        for t in range(1, Teff):
            Mx = log_delta[t-1][:, None] + np.log(self.A + 1e-12)
            psi_ptr[t] = np.argmax(Mx, axis=0)
            log_delta[t]= np.max(Mx, axis=0) + logB[t]

        path = np.zeros(Teff, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        for t in range(Teff-2, -1, -1):
            path[t] = psi_ptr[t+1, path[t+1]]

        # 还原真实状态：取上下文最后一位
        z = np.zeros(T, dtype=int)
        for i, s in enumerate(path):
            z[t0 + i] = self.last_of_ctx[s]
        # 前 t0 个（没有上下文）就用第一段的状态填充
        z[:t0] = z[t0]
        return z

    # ---------- 预测（一步） ----------
    def forecast(self, y_hist, n_steps=1):
        """
        y_hist: (L,) 历史窗口
        返回：dict，含 obs_mean/obs_var (n_steps,1)、state_proba 等（对齐你原有接口）
        """
        y = np.asarray(y_hist, dtype=float)
        L = len(y)
        if L < self.R:
            # 初始分布下的一步预测
            p = self.pi.copy()
        else:
            t0 = self.R - 1
            Teff = L - t0
            logB = np.zeros((Teff, self.S))
            for t in range(Teff):
                logB[t] = self._emission_loglik_t(y[t + t0])
            log_alpha = np.zeros((Teff, self.S))
            log_alpha[0] = np.log(self.pi + 1e-12) + logB[0]
            for t in range(1, Teff):
                la = log_alpha[t-1][:, None] + np.log(self.A + 1e-12)
                log_alpha[t] = logB[t] + logsumexp(la, axis=0)
            p = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))  # (S,)

        # 预计算混合后均值/方差（按真实状态）
        mix_mean = np.sum(self.mix_w * self.mix_mu, axis=-1)  # (K,)
        mix_var  = np.sum(self.mix_w * (self.mix_var + (self.mix_mu - mix_mean[:, None])**2), axis=-1)  # (K,)

        state_proba = np.zeros((n_steps, self.S))
        obs_mean = np.zeros((n_steps, 1))
        obs_var  = np.zeros((n_steps, 1))
        map_state = np.zeros(n_steps, dtype=int)

        for h in range(n_steps):
            p = p @ self.A
            p = p / (p.sum() + 1e-12)
            state_proba[h] = p
            map_state[h] = np.argmax(p)
            # 边缘到真实状态 k
            pm_k = p @ self.mask_last.T  # (K,)
            pm_k = pm_k / (pm_k.sum() + 1e-12)
            m = np.sum(pm_k * mix_mean)
            v = np.sum(pm_k * mix_var) + np.sum(pm_k * (mix_mean - m)**2)
            obs_mean[h, 0] = m
            obs_var[h, 0]  = v

        return {
            "state_proba": state_proba,
            "obs_mean": obs_mean,   # (n_steps,1)
            "obs_var": obs_var,
            "map_state_joint": map_state,
            "map_state_chains": [self.last_of_ctx[map_state]],  # 仅单链
        }

# =============== 多链独立 HOHMM（把每条链独立训练/推断） ===============
class IndependentHOHMMs:
    def __init__(self, K=2, num_nodes=3, order=2, mix_components=3, max_iter=80, tol=1e-5, seed=0):
        self.M = int(num_nodes)
        self.models = [
            SingleHOHMM(K=K, order=order, mix_components=mix_components,
                        max_iter=max_iter, tol=tol, seed=seed + m)
            for m in range(self.M)
        ]

    def fit(self, Y):
        """
        Y: (T, M)
        """
        Y = np.asarray(Y, dtype=float)
        for m in range(self.M):
            self.models[m].fit(Y[:, m])
        return self

    def viterbi(self, Y):
        Y = np.asarray(Y, dtype=float)
        Z = []
        for m in range(self.M):
            Z.append(self.models[m].viterbi(Y[:, m]))
        return Z  # list of length M

    def forecast(self, Y_hist, n_steps=1):
        Y_hist = np.asarray(Y_hist, dtype=float)
        M = self.M
        obs_mean = np.zeros((n_steps, M))
        obs_var  = np.zeros((n_steps, M))
        map_states = []
        for m in range(M):
            fc = self.models[m].forecast(Y_hist[:, m], n_steps=n_steps)
            obs_mean[:, m] = fc["obs_mean"][:, 0]
            obs_var[:, m]  = fc["obs_var"][:, 0]
            map_states.append(fc["map_state_chains"][0])
        return {
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "map_state_chains": map_states,
        }

# ===================== 下面是与你现有评估管线兼容的示例 =====================
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

filename = "HOHMM_" + env_name

# ==== 读数据 ====
data, data_max, data_min, data_label = prepro(env_name, None)

T = data.shape[0]

# ==== 划分 ====
train_end = int(0.8 * T)
test_beg = train_end

# ==== 训练 ====
K = 2
num_nodes = data.shape[1]
model = IndependentHOHMMs(K=K, num_nodes=num_nodes, order=2, mix_components=3,
                          max_iter=120, tol=1e-5, seed=0)
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

