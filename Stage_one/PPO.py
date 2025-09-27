import numpy as np
import torch
import torch.nn.functional as F
from utils import compute_advantage
import torch.nn as nn

entropy_coef = 0.04
temperature = 0.5


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, history_dim, action_dim):
        super(PolicyNet, self).__init__()
        # s1_dim corresponds to state["S11"] shape: (time_window, feature_dim)
        s1_dim = state_dim["S11"]
        self.num_nodes = s1_dim[0]
        self.dense1 = nn.Linear(s1_dim[0] * s1_dim[1], hidden_dim, bias=False)
        self.dense41 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.dense42 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.node_heads = nn.Linear(hidden_dim * 3, action_dim, bias=False)

        # kept (unused in forward as in original code)
        kernel_size = 2
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)

    def forward(self, input):
        x, belief, error = input["S1"], input["S3"], input["S4"]
        stateg2 = F.tanh(self.dense1(torch.flatten(x, start_dim=1)))
        beliefys = self.dense41(torch.flatten(belief, start_dim=1))
        errorys = self.dense42(torch.flatten(error, start_dim=1))
        hb = torch.concat([stateg2, beliefys, errorys], dim=-1)
        logits = self.node_heads(hb)
        logits += torch.randn_like(logits) * 1e-3
        probs_list = F.softmax(logits, dim=-1)
        return probs_list


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, history_dim, action_dim):
        super(ValueNet, self).__init__()
        s1_dim = state_dim["S11"]
        self.num_nodes = s1_dim[0]
        self.dense1 = nn.Linear(s1_dim[0] * s1_dim[1], hidden_dim, bias=False)
        self.dense41 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.dense42 = nn.Linear(history_dim * action_dim, hidden_dim, bias=False)
        self.node_heads = nn.Linear(hidden_dim * 3, 1, bias=False)
        kernel_size = 2
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)

    def forward(self, input):
        x, belief, error = input["S1"], input["S3"], input["S4"]
        stateg2 = F.tanh(self.dense1(torch.flatten(x, start_dim=1)))
        beliefys = self.dense41(torch.flatten(belief, start_dim=1))
        errorys = self.dense42(torch.flatten(error, start_dim=1))
        hb = torch.concat([stateg2, beliefys, errorys], dim=-1)
        value = self.node_heads(hb)

        return value


class PPO:

    def __init__(self, state_dim, hidden_dim, history_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, history_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, history_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs       # number of epochs to reuse the collected trajectory
        self.eps = eps             # PPO clipping range (epsilon)
        self.device = device
        self.entropy_coef = entropy_coef

    def take_action(self, state, rand=True):
        state = {
            'S1': torch.FloatTensor(state['S11']).to(self.device).unsqueeze(0),
            'S3': torch.FloatTensor(state['S3']).to(self.device).unsqueeze(0),
            'S4': torch.FloatTensor(state['S4']).to(self.device).unsqueeze(0),
        }
        probs = self.actor(state)
        if rand:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        else:
            action = torch.argmax(probs, dim=1)
        return probs.cpu().detach().numpy(), action.item()

    def update(self, transition_dict):
        states_S1 = torch.tensor(np.array(transition_dict['state_S1']), dtype=torch.float32).to(self.device)
        states_S3 = torch.tensor(np.array(transition_dict['state_S3']), dtype=torch.float32).to(self.device)
        states_S4 = torch.tensor(np.array(transition_dict['state_S4']), dtype=torch.float32).to(self.device)
        states = {
            'S1': states_S1,
            'S3': states_S3,
            'S4': states_S4
        }
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states_S1 = torch.tensor(np.array(transition_dict['nstate_S1']), dtype=torch.float32).to(self.device)
        next_states_S3 = torch.tensor(np.array(transition_dict['nstate_S3']), dtype=torch.float32).to(self.device)
        next_states_S4 = torch.tensor(np.array(transition_dict['nstate_S4']), dtype=torch.float32).to(self.device)
        next_states = {
            'S1': next_states_S1,
            'S3': next_states_S3,
            'S4': next_states_S4
        }
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        probs = self.actor(states)
        assert actions.max() < probs.shape[1], f"Action index out of range. max:{actions.max()}, action_dim:{probs.shape[1]}"

        old_log_probs = torch.log(probs.gather(1, actions)).detach()

        for _ in range(self.epochs):
            probs = self.actor(states)
            probs_smoothed = probs
            log_probs = torch.log(probs_smoothed.gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            entropy = -torch.sum(probs_smoothed * torch.log(probs_smoothed + 1e-8), dim=1).mean()
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
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
