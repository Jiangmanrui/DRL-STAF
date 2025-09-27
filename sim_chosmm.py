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
    Multi-node, state coupling + state-dependent AR + explicit duration distribution (Semi-Markov)

    Args:
        ar_dict: dict[state][node] = list of AR coefficients
        std_list: list[node] = noise std for each node
        seq_len: sequence length
        psi_list: list[node] = MxM state transition matrix for each node
        W: NxN state influence matrix
        eta: coupling strength coefficient
        duration_sampler: callable(state)->int
            Given the current state, return a dwell time >= 1
            If None, it degenerates to HMM (state may change at every step)
    """
    N = len(std_list)
    M = len(ar_dict)
    P = len(next(iter(ar_dict[0].values())))

    Y = np.zeros((seq_len, N))
    states = np.zeros((seq_len, N), dtype=int)

    # Initialize current states
    current_states = np.random.choice(M, size=N)
    # Initialize history queues
    histories = [[s] * order for s in current_states]

    # Initialize duration times
    if duration_samplers is None:
        remain_list = [1] * N
    else:
        remain_list = [duration_samplers[i][s]() for i, s in enumerate(current_states)]

    simulators = [ARIMASimulator(P) for _ in range(N)]

    for t in range(seq_len):
        new_states = current_states.copy()
        for i in range(N):
            if remain_list[i] <= 0:
                # Get historical context
                hist = histories[i][-order:]  # last "order" states
                # Transition probability tensor slice
                probs = psi_list[i][tuple(hist)]
                probs = probs / (probs.sum() + 1e-12)

                # Incorporate neighbor influence
                logits = np.log(probs + 1e-8).copy()
                for s in range(M):
                    neighbor_score = sum(W[i, j] * (current_states[j] == s) for j in range(N))
                    logits[s] += eta * neighbor_score
                probs = softmax(logits)

                # Sample new state
                new_states[i] = np.random.choice(M, p=probs)

                # Update history
                histories[i].append(new_states[i])
                if len(histories[i]) > order:
                    histories[i].pop(0)

                # Reset duration time
                if duration_samplers is not None:
                    remain_list[i] = max(1, int(duration_samplers[i][new_states[i]]()))
                else:
                    remain_list[i] = 1

            remain_list[i] -= 1  # Decrease counter

        states[t] = new_states.copy()
        current_states = new_states

        for i in range(N):
            s = current_states[i]
            ar_coeffs = ar_dict[s][i]
            Y[t, i] = simulators[i].generate_one_step(ar_coeffs, std_list[i])

    return Y, states


# ==== Parameter settings ====
N = 10   # number of nodes
M = 2    # number of states
seq_len = 10000  # sequence length

# State-dependent AR coefficients
ar_list = [[1], [-0.9]]
ar_dict = {
    s: {i: ar_list[s] for i in range(N)}
    for s in range(M)
}
std_list = [0.1] * N

# order=2, each psi[i].shape = (M,M,M)
psi_list = []
psi_dic = {}

# Node 0: strong inertia (if last two states are the same, more likely to stay)
psi0 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        psi0[a, b, :] = [0.9, 0.1] if a == b else [0.5, 0.5]

# Node 1: prefers switching to state 1 (not sensitive to history, consistent bias)
psi1 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        psi1[a, b, :] = [0.1, 0.9]

# Node 2: if history is (0->1), more likely to return to 0; otherwise prefers to stay
psi2 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        if a == 0 and b == 1:
            psi2[a, b, :] = [0.8, 0.2]
        else:
            psi2[a, b, :] = [0.7, 0.3] if a == b else [0.2, 0.8]

# Node 3: if same, more likely to switch; otherwise prefers to stay
psi3 = np.zeros((M, M, M))
for a in range(M):
    for b in range(M):
        if a == b:
            # 80% probability to switch, 20% probability to stay at b
            psi3[a, b, b] = 0.20
            psi3[a, b, 1 - b] = 0.80
        else:
            # 85% probability to stay at b, 15% probability to switch
            psi3[a, b, b] = 0.85
            psi3[a, b, 1 - b] = 0.15

# Node 4: completely random
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

# Randomly assign psi and duration samplers for N nodes
for i in range(N):
    xz = np.random.choice(range(5))
    xz2 = np.random.choice(range(10))
    psi_list.append(psi_dic[xz])
    duration_samplers.append(duration_sampler_dic[xz2])
    print(xz, xz2)

# Initialize influence matrix W
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
# ==== Data generation ====
Y, states = simulate_network_ar_with_coupled_states(ar_dict, std_list, seq_len, psi_list, W, eta,
                                                    duration_samplers=duration_samplers, order=2)

# ==== Save ====
pkl.dump((Y, states), open("./data/sim_chosmm_" + str(N) + "_" + str(seq_len) + "_g2_2_" + str(eta) + ".pkl", 'wb'))

# ==== Visualization: two rows per node group ====
time = np.arange(seq_len)

max_cols = 5
num_rows = math.ceil(N / max_cols)   # number of row groups

fig, axes = plt.subplots(num_rows * 2, max_cols, figsize=(4 * max_cols, 6 * num_rows), sharex=True)

# Ensure axes has consistent 2D shape (2*num_rows, max_cols)
axes = np.atleast_2d(axes)

for i in range(N):
    col = i % max_cols         # which column for this node
    row_group = i // max_cols  # which group of rows
    row_obs = row_group * 2    # observations in even rows
    row_state = row_group * 2 + 1  # states in odd rows

    # Upper row: observations
    axes[row_obs, col].plot(time, Y[:, i], color='tab:blue')
    axes[row_obs, col].set_title(f'Node {i} - Observation')
    axes[row_obs, col].set_ylabel('Value')
    axes[row_obs, col].grid(True)

    # Lower row: states
    axes[row_state, col].step(time, states[:, i], color='tab:orange', where='post')
    axes[row_state, col].set_title(f'Node {i} - State')
    axes[row_state, col].set_xlabel('Time')
    axes[row_state, col].set_ylabel('State')
    axes[row_state, col].set_yticks([0, 1])
    axes[row_state, col].grid(True)

# Remove extra subplots if N is not a multiple of max_cols
for j in range(N, num_rows * max_cols):
    col = j % max_cols
    row_group = j // max_cols
    fig.delaxes(axes[row_group*2, col])
    fig.delaxes(axes[row_group*2+1, col])

plt.suptitle(f"Coupled Hidden States & State-dependent AR: N={N}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
