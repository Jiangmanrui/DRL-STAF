import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ARIMASimulator:
    def __init__(self, max_lag):
        self.data = []
        self.max_lag = max_lag

    def generate_one_step(self, ar, std):
        noise = np.random.randn() * std
        value = noise
        for j in range(len(ar)):
            if len(self.data) > j:
                value += ar[j] * self.data[-(j + 1)]
        self.data.append(value)
        return value


def simulate_network_ar_with_coupled_states(ar_dict, std_list, seq_len, psi_list, W, eta=1.0, duration_samplers=None,
                                            order=1):
    """
    多节点、状态耦合 + 状态依赖AR + 显式滞留时间分布（Semi-Markov）

    参数说明：
        ar_dict: dict[state][node] = AR 系数列表
        std_list: list[node] = 每个节点的噪声标准差
        seq_len: 序列长度
        psi_list: list[node] = 每个节点的 MxM 状态转移矩阵
        W: NxN 状态影响矩阵
        eta: 状态耦合强度系数
        duration_sampler: callable(state)->int
            给定当前状态，返回一个 >=1 的停留时间
            如果为 None，则退化为 HMM（每步都可能转移）
    """
    N = len(std_list)
    M = len(ar_dict)
    P = len(next(iter(ar_dict[0].values())))

    Y = np.zeros((seq_len, N))
    states = np.zeros((seq_len, N), dtype=int)

    # 初始化当前状态
    current_states = np.random.choice(M, size=N)
    # 初始化历史队列
    histories = [[s] * order for s in current_states]

    # 初始化停留时间
    if duration_samplers is None:
        remain_list = [1] * N
    else:
        remain_list = [duration_samplers[i][s]() for i, s in enumerate(current_states)]

    simulators = [ARIMASimulator(P) for _ in range(N)]

    for t in range(seq_len):
        new_states = current_states.copy()
        for i in range(N):
            if remain_list[i] <= 0:
                # 获取历史上下文
                hist = histories[i][-order:]  # 最近 order 个状态
                # 转移概率张量切片
                probs = psi_list[i][tuple(hist)]
                probs = probs / (probs.sum() + 1e-12)

                # 融合邻居影响
                logits = np.log(probs + 1e-8).copy()
                for s in range(M):
                    neighbor_score = sum(W[i, j] * (current_states[j] == s) for j in range(N))
                    logits[s] += eta * neighbor_score
                probs = softmax(logits)

                # 采样新状态
                new_states[i] = np.random.choice(M, p=probs)

                # 更新历史
                histories[i].append(new_states[i])
                if len(histories[i]) > order:
                    histories[i].pop(0)

                # 重置停留时间
                if duration_samplers is not None:
                    remain_list[i] = max(1, int(duration_samplers[i][new_states[i]]()))
                else:
                    remain_list[i] = 1

            remain_list[i] -= 1  # 计数器减一

        states[t] = new_states.copy()
        current_states = new_states

        for i in range(N):
            s = current_states[i]
            ar_coeffs = ar_dict[s][i]
            Y[t, i] = simulators[i].generate_one_step(ar_coeffs, std_list[i])

    return Y, states


# ==== 参数设定 ====
N = 10  # 节点数
M = 2  # 状态数
seq_len = 10000  # 时间长度

# 状态-AR 系数设定
ar_list = [[1], [-0.9]]
ar_dict = {
    s: {i: ar_list[s] for i in range(N)}
    for s in range(M)
}
std_list = [0.1] * N

# # 每节点独立状态转移矩阵 psi_list[i]
# psi_list = [
#     np.array([[0.95, 0.05],
#               [0.10, 0.90]]),  # 节点0
#     np.array([[0.90, 0.10],
#               [0.05, 0.95]]),  # 节点1
#     np.array([[0.92, 0.08],
#               [0.08, 0.92]])  # 节点2
# ]

# order=2, 每个 psi[i].shape = (M,M,M)
psi_list = []
psi_dic = {}
# 节点0：强惯性（前两段相同就更可能继续保持）
psi0 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        psi0[a, b, :] = [0.9, 0.1] if a == b else [0.5, 0.5]

# 节点1：喜欢切换到状态1（不太看历史，一致偏好1）
psi1 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        psi1[a, b, :] = [0.1, 0.9]

# 节点2：若历史为(0->1)则更易回到0，否则偏向保持
psi2 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        if a == 0 and b == 1:
            psi2[a, b, :] = [0.8, 0.2]
        else:
            psi2[a, b, :] = [0.7, 0.3] if a == b else [0.2, 0.8]

# 节点3：相同则更容易变，否则保持
psi3 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        if a == b:
            # 80% 概率切换到另一状态，20% 概率留在 b
            psi3[a, b, b] = 0.20
            psi3[a, b, 1 - b] = 0.80
        else:
            # 85% 概率保持 b，15% 概率去另一个
            psi3[a, b, b] = 0.85
            psi3[a, b, 1 - b] = 0.15

# 节点4：完全随机
psi4 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        psi4[a, b, :] = [0.5, 0.5]

psi_dic[0] = psi0
psi_dic[1] = psi1
psi_dic[2] = psi2
psi_dic[3] = psi3
psi_dic[4] = psi4

# N=3
# duration_samplers = [
#     [lambda: np.random.geometric(0.01),
#      lambda: 1 + np.random.poisson(250)],
#     [lambda: np.random.geometric(0.001),
#      lambda: 1 + np.random.poisson(20)],
#     [lambda: 200,
#      lambda: np.random.geometric(0.0025)]
# ]

duration_sampler_dic = {0: [lambda: np.random.geometric(0.01),
                            lambda: 1 + np.random.poisson(250)],
                        1: [lambda: np.random.geometric(0.001),
                            lambda: 1 + np.random.poisson(20)],
                        2: [lambda: 200,
                            lambda: np.random.geometric(0.0025)],
                        3: [lambda: 1 + np.random.poisson(100),
                            lambda: 1 + np.random.poisson(100)],
                        4: [lambda: np.random.geometric(0.01),
                            lambda: np.random.geometric(0.005)],
                        5: [lambda: np.random.geometric(0.01),
                            lambda: 1 + np.random.poisson(250)],
                        6: [lambda: np.random.geometric(0.001),
                            lambda: 1 + np.random.poisson(20)],
                        7: [lambda: 200,
                            lambda: np.random.geometric(0.0025)],
                        8: [lambda: 1 + np.random.poisson(100),
                            lambda: 1 + np.random.poisson(100)],
                        9: [lambda: np.random.geometric(0.01),
                            lambda: np.random.geometric(0.005)], }

duration_samplers = []

# 10
for i in range(N):
    xz = np.random.choice(range(5))
    xz2 = np.random.choice(range(10))
    psi_list.append(psi_dic[xz])
    duration_samplers.append(duration_sampler_dic[xz2])
    print(xz,xz2)

# W = np.array([
#     [0, 1, 0],
#     [1, 0, 1],
#     [0, 1, 0]
# ])

W = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i != j:
            W[i, j] = np.random.choice([0, 1], p=[0.5, 0.5])
np.save("./data/sim_chosmm_W_" + str(N) + ".npy", W)
print(W)

eta = 0.2
# ==== 数据生成 ====
Y, states = simulate_network_ar_with_coupled_states(ar_dict, std_list, seq_len, psi_list, W, eta,
                                                    duration_samplers=duration_samplers, order=2)

# ==== 保存 ====
pkl.dump((Y, states), open("./data/sim_chosmm_" + str(N) + "_" + str(seq_len) + "_g2_2_" + str(eta) + ".pkl", 'wb'))

# ==== 可视化：两行N列子图 ====
time = np.arange(seq_len)

max_cols = 5
num_rows = math.ceil(N / max_cols)   # 需要多少组行

fig, axes = plt.subplots(num_rows * 2, max_cols, figsize=(4 * max_cols, 6 * num_rows), sharex=True)

# 统一 axes 维度为二维 (2*num_rows, max_cols)
axes = np.atleast_2d(axes)

for i in range(N):
    col = i % max_cols         # 当前节点放在哪一列
    row_group = i // max_cols  # 当前节点属于第几组
    row_obs = row_group * 2    # 观测在偶数行
    row_state = row_group * 2 + 1  # 状态在奇数行

    # 上行：观测值
    axes[row_obs, col].plot(time, Y[:, i], color='tab:blue')
    axes[row_obs, col].set_title(f'Node {i} - Observation')
    axes[row_obs, col].set_ylabel('Value')
    axes[row_obs, col].grid(True)

    # 下行：状态序列
    axes[row_state, col].step(time, states[:, i], color='tab:orange', where='post')
    axes[row_state, col].set_title(f'Node {i} - State')
    axes[row_state, col].set_xlabel('Time')
    axes[row_state, col].set_ylabel('State')
    axes[row_state, col].set_yticks([0, 1])
    axes[row_state, col].grid(True)

# 删除多余子图（如果 N 不是 max_cols 的倍数）
for j in range(N, num_rows * max_cols):
    col = j % max_cols
    row_group = j // max_cols
    fig.delaxes(axes[row_group*2, col])
    fig.delaxes(axes[row_group*2+1, col])

plt.suptitle(f"Coupled Hidden States & State-dependent AR: N={N}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
