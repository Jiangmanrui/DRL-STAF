# HSMM.py
import numpy as np
from math import lgamma
import os
import pandas as pd
import matplotlib.pyplot as plt
from PySide6.QtWebEngineCore import qWebEngineVersion

from DATAPREPRO import prepro
from tqdm import tqdm

def logsumexp_np(a, axis=None, keepdims=False):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + 1e-300)
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)

def gaussian_logpdf(x, mu, var):
    eps = 1e-12
    var = np.maximum(var, eps)
    return -0.5 * (np.log(2*np.pi*var) + (x - mu)**2 / var)

class _GMM1D:
    """ 1D GMM（用于每条链、每个状态的发射） """
    def __init__(self, n_components=1, rng=None):
        self.C = int(n_components)
        self.w = None   # (C,)
        self.mu = None  # (C,)
        self.var = None # (C,)
        self.rng = np.random.default_rng(None if rng is None else rng)

    def init_from_data(self, y, Kmeans_centers=None):
        y = np.asarray(y).ravel()
        if self.C == 1:
            self.w = np.array([1.0])
            self.mu = np.array([np.mean(y)])
            self.var = np.array([np.var(y) + 1e-2])
        else:
            if Kmeans_centers is None:
                centers = np.linspace(np.min(y), np.max(y), self.C)
            else:
                centers = np.asarray(Kmeans_centers).ravel()
                if centers.size != self.C:
                    centers = np.linspace(np.min(y), np.max(y), self.C)
            self.w = np.full(self.C, 1.0 / self.C)
            self.mu = centers.copy()
            self.var = np.full(self.C, np.var(y)/(self.C) + 1e-2)

    def loglik(self, y):
        """ 返回 log \sum_c w_c N(y|mu_c, var_c) """
        y = np.asarray(y).ravel()
        lw = np.log(self.w + 1e-300)[None, :]           # (1,C)
        lN = gaussian_logpdf(y[:, None], self.mu[None, :], self.var[None, :])  # (T,C)
        return logsumexp_np(lw + lN, axis=1)            # (T,)

    def em_step(self, y, resp=None, n_iter=1):
        """
        y: (T,) 训练样本；resp: (T,) 作为该状态的权重，可选（Viterbi-EM 时可不传）
        执行 n_iter 次局部 EM，更新 w, mu, var
        """
        y = np.asarray(y).ravel()
        T = y.shape[0]
        for _ in range(n_iter):
            # E: 组件后验
            lw = np.log(self.w + 1e-300)[None, :]               # (1,C)
            lN = gaussian_logpdf(y[:, None], self.mu[None, :], self.var[None, :])  # (T,C)
            log_post = lw + lN
            log_post = log_post - logsumexp_np(log_post, axis=1, keepdims=True)
            Q = np.exp(log_post)                                 # (T,C)

            if resp is not None:
                R = resp[:, None] * Q
            else:
                R = Q

            Nk = np.sum(R, axis=0) + 1e-12  # (C,)
            self.w = Nk / (np.sum(Nk) + 1e-12)
            self.mu = np.sum(R * y[:, None], axis=0) / Nk
            self.var = np.maximum(np.sum(R * (y[:, None] - self.mu[None, :])**2, axis=0) / Nk, 1e-6)


class HSMMChain:
    """
    单条链的 HSMM（显式停留时间分布）。
    训练用 Viterbi-EM（分段解码 + MLE 重估）。
    """
    def __init__(self, K=2, C=1, Dmax=200, duration_model='poisson', seed=0):
        self.K = int(K)
        self.C = int(C)
        self.Dmax = int(Dmax)
        self.duration_model = duration_model  # 'poisson' or 'geometric'
        self.rng = np.random.default_rng(seed)

        # 初值
        self.pi = np.full(self.K, 1.0/self.K)  # 初始状态分布（按段）
        self.A = np.full((self.K, self.K), 1.0/self.K)
        self.gmms = [ _GMM1D(self.C, rng=self.rng) for _ in range(self.K) ]

        # 时长参数（K,）
        if self.duration_model == 'poisson':
            self.lambda_k = np.full(self.K, 50.0)  # 可以在 fit 中重估
        else:  # geometric, pmf(d)=p(1-p)^{d-1}, mean=1/p
            self.p_k = np.full(self.K, 1/50.0)

        # 截断 pmf（K,Dmax）
        self.dur_pmf = None

    # -------- durations ----------
    def _build_duration_pmf(self):
        """ 根据当前参数在 1..Dmax 上构造（截断）时长 pmf，shape=(K,Dmax) """
        D = self.Dmax
        d = np.arange(1, D+1, dtype=float)  # (D,)
        if self.duration_model == 'poisson':
            lam = np.maximum(self.lambda_k, 1e-8)  # (K,)
            # log pmf(d) = -lam + d log lam - log(d!)
            log_fact = np.array([lgamma(int(i)+1) for i in d])  # (D,)
            logpmf = (-lam[:, None]
                      + np.log(lam[:, None]) * d[None, :]
                      - log_fact[None, :])  # (K,D)
        else:
            # geometric on {1,2,...}: pmf(d)=p(1-p)^{d-1}
            p = np.clip(self.p_k, 1e-8, 1-1e-8)  # (K,)
            logpmf = (np.log(p)[:, None] + (d[None, :]-1.0)*np.log(1.0-p)[:, None])

        # 截断重归一化
        logpmf = logpmf - logsumexp_np(logpmf, axis=1, keepdims=True)
        pmf = np.exp(logpmf)
        pmf = np.maximum(pmf, 1e-300)
        self.dur_pmf = pmf  # (K,D)

    # -------- segment Viterbi --------
    def _segment_loglik(self, y, emit_loglik=None):
        """
        预计算每个状态 k 的分段对数似然：
        seg_ll[end, d, k] = sum_{t=end-d+1..end} log b_k(y_t)
        """
        T = y.shape[0]
        K = self.K
        D = self.Dmax
        if emit_loglik is None:
            emit_loglik = np.zeros((T, K))
            for k in range(K):
                emit_loglik[:, k] = self.gmms[k].loglik(y)
        csum = np.vstack([np.zeros((1, K)), np.cumsum(emit_loglik, axis=0)])  # (T+1,K)
        max_d = min(D, T)
        seg_ll = np.full((T, max_d, K), -np.inf)
        for end in range(T):
            maxd = min(max_d, end+1)
            # sum of logs on [end-d+1, end] → csum[end+1]-csum[end+1-d]
            block = csum[end+1, :][None, :] - csum[end+1 - np.arange(1, maxd+1), :]
            seg_ll[end, :maxd, :] = block  # (maxd,K)
        return seg_ll, emit_loglik

    def viterbi_decode(self, y):
        """
        分段 Viterbi：返回
        - z: (T,) 时刻级别的最优路径（按状态编号）
        - segments: list[(start, end, k)] 的段级别路径
        """
        y = np.asarray(y).ravel()
        T, K, D = y.shape[0], self.K, self.Dmax
        self._build_duration_pmf()
        seg_ll, _ = self._segment_loglik(y)  # (T,≤D,K)

        # DP：dp[t,k] 最佳路径以 state k 在 t 处结束
        dp = np.full((T, K), -np.inf)
        back_ptr = [[None for _ in range(K)] for _ in range(T)]  # 记录 (t_prev, k_prev, d)

        # 初始化：第一段
        for t in range(T):
            maxd = min(D, t+1)
            for k in range(K):
                best = -np.inf
                best_bp = None
                for d in range(1, maxd+1):
                    t0 = t - d + 1
                    score = np.log(self.pi[k] + 1e-300) + np.log(self.dur_pmf[k, d-1]) \
                            + seg_ll[t, d-1, k]
                    if t0 == 0:
                        # 第一段，无前驱
                        prev = 0.0
                    else:
                        # 需要从某个 j 转移到 k
                        prev = np.max(dp[t0-1, :] + np.log(self.A[:, k] + 1e-300))
                    tot = score + prev
                    if tot > best:
                        best = tot
                        if t0 == 0:
                            best_bp = (None, None, d)
                        else:
                            j = int(np.argmax(dp[t0-1, :] + np.log(self.A[:, k] + 1e-300)))
                            best_bp = (t0-1, j, d)
                dp[t, k] = best
                back_ptr[t][k] = best_bp

        # 终止：选择 t=T-1 的最佳 k
        t = T-1
        k = int(np.argmax(dp[t, :]))
        segments = []
        while True:
            t_prev, k_prev, d = back_ptr[t][k]
            start = t - d + 1
            segments.append((start, t, k))
            if t_prev is None:
                break
            t = t_prev
            k = k_prev
        segments.reverse()

        # 展开到逐时刻路径
        z = np.zeros(T, dtype=int)
        for s, e, k in segments:
            z[s:e+1] = k
        return z, segments

    # -------- Viterbi-EM 训练 --------
    def fit(self, y, n_iter=50, gmm_em_iter=2):
        """
        y: (T,)
        n_iter: Viterbi-EM 迭代次数
        """
        y = np.asarray(y).ravel()
        T = y.shape[0]

        # 初始化发射
        for k in range(self.K):
            # 先粗分配：按值分箱
            cuts = np.quantile(y, np.linspace(0, 1, self.K+1))
            mask = (y >= cuts[k]) & (y <= cuts[k+1])
            yk = y[mask] if np.any(mask) else y
            centers = np.linspace(np.percentile(yk, 10), np.percentile(yk, 90), self.C)
            self.gmms[k].init_from_data(yk, Kmeans_centers=centers)

        # 初始化转移为“弱自环”
        self.A = np.full((self.K, self.K), 1e-3)
        np.fill_diagonal(self.A, 1.0)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.full(self.K, 1.0/self.K)

        for _ in range(n_iter):
            # 1) Viterbi 解码得到段
            z, segments = self.viterbi_decode(y)

            # 2) 重估 pi, A（段级别）
            if len(segments) > 0:
                self.pi = np.zeros(self.K)
                self.pi[segments[0][2]] = 1.0
                # 段转移计数
                counts = np.full((self.K, self.K), 1e-6)
                for i in range(1, len(segments)):
                    j = segments[i-1][2]
                    k = segments[i][2]
                    counts[j, k] += 1.0
                self.A = counts / counts.sum(axis=1, keepdims=True)

            # 3) 重估时长参数
            lengths_by_k = [[] for _ in range(self.K)]
            for (s, e, k) in segments:
                lengths_by_k[k].append(e - s + 1)
            for k in range(self.K):
                if len(lengths_by_k[k]) == 0:
                    continue
                mean_d = np.mean(lengths_by_k[k])
                if self.duration_model == 'poisson':
                    self.lambda_k[k] = max(1.0, float(mean_d))
                else:
                    self.p_k[k] = np.clip(1.0 / mean_d, 1e-6, 1-1e-6)

            # 4) 重估 GMM（按状态聚合样本）
            for k in range(self.K):
                yk = y[z == k]
                if yk.size == 0:
                    continue
                self.gmms[k].em_step(yk, n_iter=gmm_em_iter)

        return self

    # -------- 近似 forecast（一步）--------
    def forecast_one_step(self, y_hist):
        """
        近似一步预测：将 HSMM 近似成等价 HMM（自环概率由 mean duration 推出），
        然后用该 HMM 做一步预测的观测均值/方差（GMM 混合后）。
        返回 mean, var
        """
        y = np.asarray(y_hist).ravel()
        # 1) 等价 HMM 自环概率
        if self.duration_model == 'poisson':
            mean_d = np.maximum(self.lambda_k, 1.0)
        else:
            mean_d = 1.0 / np.maximum(self.p_k, 1e-6)
        p_self = np.clip(1.0 - 1.0/mean_d, 1e-6, 1-1e-6)  # (K,)

        A_eff = self.A.copy()
        # 归一化去除自环的出环概率
        for k in range(self.K):
            out = A_eff[k].copy()
            out[k] = 0.0
            s = out.sum()
            if s < 1e-12:
                A_eff[k] = 0.0
                A_eff[k, k] = 1.0
            else:
                out = out / s
                A_eff[k] = (1.0 - p_self[k]) * out
                A_eff[k, k] = p_self[k]

        # 2) 用最后一段状态作近似过滤（更简单稳健：直接用全序列 Viterbi 的末状态）
        #   你也可以改成用一个短窗口做 HMM 前向过滤
        z_vit, segs = self.viterbi_decode(y)
        kT = int(z_vit[-1])

        # 3) 下一步状态分布
        p_next = A_eff[kT]  # (K,)

        # 4) 发射（GMM 的均值/方差混合）
        mix_mean = np.array([np.sum(g.w * g.mu) for g in self.gmms])        # (K,)
        mix_var  = np.array([np.sum(g.w * (g.var + (g.mu - np.sum(g.w*g.mu))**2)) for g in self.gmms])  # (K,)

        mean = float(np.sum(p_next * mix_mean))
        var  = float(np.sum(p_next * (mix_var + (mix_mean - mean)**2)))
        return mean, var


class IndependentHSMMs:
    """
    多条链的独立 HSMM：每条链一个 HSMMChain，互不耦合。
    """
    def __init__(self, K=2, num_nodes=3, mix_components=1, Dmax=200, duration_model='poisson', seed=0):
        self.M = int(num_nodes)
        self.K = int(K)
        self.C = int(mix_components)
        self.Dmax = int(Dmax)
        self.duration_model = duration_model
        self.seed = seed
        # 每条链一个 HSMMChain
        self.chains = [HSMMChain(K=K, C=self.C, Dmax=self.Dmax,
                                 duration_model=self.duration_model,
                                 seed=seed + m) for m in range(self.M)]

    def fit(self, Y, n_iter=50, gmm_em_iter=2):
        """
        Y: (T, M) 多链观测
        对每条链独立训练
        """
        Y = np.asarray(Y, dtype=float)
        T, M = Y.shape
        assert M == self.M, "num_nodes 与数据列数不一致"
        for m in range(M):
            self.chains[m].fit(Y[:, m], n_iter=n_iter, gmm_em_iter=gmm_em_iter)
        return self

    def viterbi(self, Y):
        """
        返回 list，长度 M，每条是 (T,) 的最优状态路径
        """
        Y = np.asarray(Y, dtype=float)
        T, M = Y.shape
        paths = []
        for m in range(M):
            z, _ = self.chains[m].viterbi_decode(Y[:, m])
            paths.append(z)
        return paths

    def forecast(self, Y_hist, n_steps=1):
        """
        近似预测：逐链独立地做一步预测；若 n_steps>1，滚动重复（近似）
        返回 dict:
          - obs_mean: (n_steps, M)
          - obs_var:  (n_steps, M)
        """
        Y_hist = np.asarray(Y_hist, dtype=float)
        T, M = Y_hist.shape
        means = np.zeros((n_steps, M))
        vars_ = np.zeros((n_steps, M))

        # 简单滚动：不将预测均值回填到序列（保持近似与稳健）
        for h in range(n_steps):
            for m in range(M):
                mu, va = self.chains[m].forecast_one_step(Y_hist[:, m])
                means[h, m] = mu
                vars_[h, m] = va
        return {"obs_mean": means, "obs_var": vars_}




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

filename = "HSMM_" + env_name

# ==== 读数据 ====
data, data_max, data_min, data_label = prepro(env_name, None)

T = data.shape[0]

# ==== 划分 ====
train_end = int(0.8 * T)
test_beg = train_end

# ==== 训练 ====
K = 2
num_nodes = data.shape[1]
model = IndependentHSMMs(K=2, num_nodes=data.shape[1], mix_components=3,
                         Dmax=200, duration_model='poisson', seed=0)
model.fit(data[:train_end], n_iter=60, gmm_em_iter=2)

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
