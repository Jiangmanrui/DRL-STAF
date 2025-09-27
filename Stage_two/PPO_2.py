import numpy as np
import torch
import torch.nn.functional as F
from utils import compute_advantage2 as compute_advantage
import torch.nn as nn

entropy_coef = 0.05
temperature = 0.5


class GraphAttentionLayer(nn.Module):
    def __init__(self, n_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.n_features = n_features
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat

        self.a = nn.Parameter(torch.empty(2 * n_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh: [B, T, N, F]
        Wh1 = torch.matmul(Wh, self.a[:self.n_features, :])  # [B, T, N, 1]
        Wh2 = torch.matmul(Wh, self.a[self.n_features:, :])  # [B, T, N, 1]
        e = Wh1 + Wh2.transpose(-2, -1)  # [B, T, N, N]
        return self.leakyrelu(e)

    def forward(self, h_prime, adj):
        # h_prime: [B, T, N, F], adj: [N, N]
        B, T, N, feature_dim = h_prime.shape
        assert adj.shape == (N, N)

        e = self._prepare_attentional_mechanism_input(h_prime)  # [B, T, N, N]

        adj_expanded = adj.unsqueeze(0).unsqueeze(0).expand(B, T, N, N)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        out = torch.matmul(attention, h_prime)  # [B, T, N, F]
        return F.elu(out) if self.concat else out


class GAT(nn.Module):
    def __init__(self, n_features, dropout, alpha, nheads, type="concat"):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.type = type

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_features, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])

    def forward(self, x, adj):
        # x: [B, T, N, F], adj: [N, N]
        x = F.dropout(x, self.dropout, training=self.training)
        if self.type == "concat":
            x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        else:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=0).mean(dim=0)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class ResidualGAT(nn.Module):
    def __init__(
            self,
            in_feats: int,
            dropout: float,
            alpha: float,
            nheads: int,
            merge: str = "concat"
    ):

        super().__init__()
        self.in_feats = in_feats
        self.nheads = nheads
        self.merge = merge

        # Traditional GAT layer
        self.gat = GAT(in_feats, dropout, alpha, nheads, merge)


        out_feats = in_feats * nheads if merge == "concat" else in_feats
        self.project = nn.Linear(out_feats, in_feats)

    def forward(self, x, adj):
        """
        x:   [B, T, N, in_feats]
        adj: [N, N]
        """
        # 1) GAT multi-heads
        h = self.gat(x, adj)  # [B, T, N, out_feats]

        # 2) Back to the original feature dimension
        h = self.project(h)  # [B, T, N, in_feats]

        # 3) Residual connection
        return F.elu(h + x)  # [B, T, N, in_feats]


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, history_dim, action_dim, net, device):
        super(PolicyNet, self).__init__()

        s1_dim = state_dim["S1"]
        self.num_nodes = s1_dim[0]
        self.dense2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.dense3 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.dense4 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.node_heads = nn.Linear(hidden_dim, action_dim, bias=False)

        self.network = torch.tensor(net, dtype=torch.float32).to(device)
        self.gat = ResidualGAT(hidden_dim, 0.2, 0.2, 2, 'concat')

        self.alpha_node = nn.Parameter(torch.ones(self.num_nodes))

    def forward(self, input):
        x, belief, error, fresult = input["S1"], input["S3"], input["S4"], input["S5"]
        fresult = fresult.squeeze(2)
        belief = belief.reshape(belief.shape[0], belief.shape[1], -1)
        error = error.reshape(error.shape[0], error.shape[1], -1)
        beliefys = self.dense3(belief)
        errorys = self.dense4(error)
        hb = torch.concat([beliefys, errorys], dim=-1)
        hb_embed = self.dense2(hb)
        hb_gat = self.gat(hb_embed.unsqueeze(1), self.network).squeeze(1)

        logits = self.node_heads(hb_gat)
        logits += torch.randn_like(logits) * 1e-3

        trust = torch.sigmoid(self.alpha_node).view(1, -1, 1)  # [1,N,1]
        f_adj = fresult * trust

        logits = logits + f_adj
        probs = F.softmax(logits, dim=-1)
        probs_list = []
        for n in range(self.num_nodes):
            probs_list.append(probs[:, n, :])
        return probs_list


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, history_dim, action_dim, net, device):
        super(ValueNet, self).__init__()
        s1_dim = state_dim["S1"]
        self.num_nodes = s1_dim[0]
        self.dense2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.dense3 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.dense4 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.node_heads = nn.Linear(hidden_dim, action_dim, bias=False)
        self.value_heads = nn.Linear(action_dim * 2, 1, bias=False)

        self.network = torch.tensor(net, dtype=torch.float32).to(device)
        self.gat = ResidualGAT(hidden_dim, 0.2, 0.2, 2, 'concat')

        self.alpha_node = nn.Parameter(torch.ones(self.num_nodes))

    def forward(self, input):
        x, belief, error, fresult = input["S1"], input["S3"], input["S4"], input["S5"]
        belief = belief.reshape(belief.shape[0], belief.shape[1], -1)
        error = error.reshape(error.shape[0], error.shape[1], -1)
        fresult = fresult.squeeze(2)
        beliefys = self.dense3(belief)
        errorys = self.dense4(error)
        hb = torch.concat([beliefys, errorys], dim=-1)
        hb_embed = self.dense2(hb)
        hb_gat = self.gat(hb_embed.unsqueeze(1), self.network).squeeze(1)
        logits = self.node_heads(hb_gat)

        trust = torch.sigmoid(self.alpha_node).view(1, -1, 1)  # [1,N,1]
        f_adj = fresult * trust

        hb_2 = torch.concat([f_adj, logits], dim=-1)
        value = self.value_heads(hb_2)
        value = value.squeeze(-1)
        return value


class PPO:
    def __init__(self, state_dim, hidden_dim, history_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, net, device):
        self.actor = PolicyNet(state_dim, hidden_dim, history_dim, action_dim, net, device).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, history_dim, action_dim, net, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.entropy_coef = entropy_coef

    def take_action(self, state, rand=True):
        state = {
            'S1': torch.FloatTensor(state['S1']).to(self.device).unsqueeze(0),
            'S3': torch.FloatTensor(state['S3']).to(self.device).unsqueeze(0),
            'S4': torch.FloatTensor(state['S4']).to(self.device).unsqueeze(0),
            'S5': torch.FloatTensor(state['S5']).to(self.device).unsqueeze(0),
        }
        probs_lists = self.actor(state)
        actions = []
        for probs in probs_lists:
            action_dist = torch.distributions.Categorical(probs)
            if rand:
                action = action_dist.sample()
            else:
                action = torch.argmax(probs, dim=1)
            actions.append(action)

        probs_stack = torch.stack(probs_lists, dim=1).squeeze(0).unsqueeze(1).cpu().detach().numpy()
        actions_stack = torch.stack(actions, dim=1).squeeze(0).cpu().detach().numpy()

        return probs_stack, actions_stack

    def update(self, transition_dict):
        states_S1 = torch.tensor(transition_dict['state_S1'], dtype=torch.float32).to(self.device)
        states_S3 = torch.tensor(transition_dict['state_S3'], dtype=torch.float32).to(self.device)
        states_S4 = torch.tensor(transition_dict['state_S4'], dtype=torch.float32).to(self.device)
        states_S5 = torch.tensor(transition_dict['state_S5'], dtype=torch.float32).to(self.device)
        states = {
            'S1': states_S1,
            'S3': states_S3,
            'S4': states_S4,
            'S5': states_S5
        }
        actions = torch.tensor(transition_dict['actions']).to(self.device)  # [B, N]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).to(self.device)  # [B, N]
        next_states_S1 = torch.tensor(transition_dict['nstate_S1'], dtype=torch.float32).to(self.device)
        next_states_S3 = torch.tensor(transition_dict['nstate_S3'], dtype=torch.float32).to(self.device)
        next_states_S4 = torch.tensor(transition_dict['nstate_S4'], dtype=torch.float32).to(self.device)
        next_states_S5 = torch.tensor(transition_dict['nstate_S5'], dtype=torch.float32).to(self.device)
        next_states = {
            'S1': next_states_S1,
            'S3': next_states_S3,
            'S4': next_states_S4,
            'S5': next_states_S5
        }
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).to(self.device)
        if dones.ndim == 1:
            dones = dones.unsqueeze(1).expand_as(rewards)

        with torch.no_grad():
            value_detached = self.critic(states)  # [B, N]
            next_value_detached = self.critic(next_states)  # [B, N]
            td_target = rewards + self.gamma * next_value_detached * (1 - dones)  # [B, N]
            td_delta = td_target - value_detached  # [B, N]
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            probs_list = self.actor(states)
            old_log_probs = torch.stack([
                torch.log(probs_list[i].gather(1, actions[:, i].unsqueeze(1)).clamp(min=1e-8)).squeeze(1)
                for i in range(len(probs_list))
            ], dim=1)  # [B, N]

        # === 2. Multiple rounds of PPO training ===
        for _ in range(self.epochs):
            probs_list = self.actor(states)
            value = self.critic(states)  # [B, N]

            log_probs = torch.stack([
                torch.log(probs_list[i].gather(1, actions[:, i].unsqueeze(1)).clamp(min=1e-8)).squeeze(1)
                for i in range(len(probs_list))
            ], dim=1)  # [B, N]

            entropy = torch.stack([
                -torch.sum(probs_list[i] * torch.log(probs_list[i] + 1e-8), dim=1)
                for i in range(len(probs_list))
            ], dim=1).mean()

            ratio = torch.exp(log_probs - old_log_probs)  # [B, N]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            critic_loss = F.mse_loss(value, td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{name}_critic.pth')

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load(f'{directory}/{name}_actor.pth', map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(
            torch.load(f'{directory}/{name}_critic.pth', map_location=lambda storage, loc: storage))
