import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import compute_advantage
from torch.utils.data import DataLoader, TensorDataset, random_split

class PNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_states):
        super(PNet, self).__init__()
        nhead = 4
        num_layers = 1
        s1_dim = input_dim["S1"]
        self.num_state = num_states
        self.input_proj = nn.Linear(s1_dim[1], hidden_dim)

        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
                                           batch_first=True),
                num_layers=num_layers
            )
            for _ in range(num_states)
        ])

        self.output_proj = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_states)
        ])

    def forward(self, x, action):
        x_embed = self.input_proj(x)
        preds = []
        prob = action
        for k in range(self.num_state):
            encoded = self.encoders[k](x_embed)
            pred = self.output_proj[k](encoded[:, -1, :])
            preds.append(pred)
        preds = torch.concat(preds, dim=1).unsqueeze(-1)
        out = torch.matmul(prob, preds)
        return out, preds


class PNetbase(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_states):
        super(PNetbase, self).__init__()
        nhead = 4
        num_layers = 1
        s1_dim = input_dim["S1"]
        self.num_state = num_states
        self.input_proj = nn.Linear(s1_dim[1], hidden_dim)

        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
                                       batch_first=True),
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x_embed = self.input_proj(x)
        encoded = self.encoders(x_embed)
        pred = self.output_proj(encoded[:, -1, :]).unsqueeze(1)
        return pred


class PREDM:
    def __init__(self, input_dim, pred_dim, hidden_dim, num_states, lr, epochs, pat, tau, param, device):
        self.pred = PNet(input_dim, hidden_dim, pred_dim, num_states).to(device)
        self.predbase = PNetbase(input_dim, hidden_dim, pred_dim, num_states).to(device)
        self.target_pred = PNet(input_dim, hidden_dim, pred_dim, num_states).to(device)
        self.target_pred.load_state_dict(self.pred.state_dict())  # 初始化为当前 pred 的参数
        self.tau = tau  # 软更新系数，可调
        self.param = param
        self.optimizer = torch.optim.Adam(self.pred.parameters(),
                                          lr=lr)
        self.optimizerbase = torch.optim.Adam(self.predbase.parameters(),
                                              lr=lr)
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.patience = pat
        self.device = device
        self.criterion = nn.MSELoss()
        self.num_states = num_states
        self.use_target_pred = False

    def updatebase(self, transition_dict):
        states_S1 = torch.tensor(np.array(transition_dict['pred_state_S1']), dtype=torch.float32).to(self.device)
        inputdata = {
            'S1': states_S1,
        }
        outputdata = torch.tensor(np.array(transition_dict['pred_target']), dtype=torch.float32).to(self.device)

        dataset = TensorDataset(inputdata["S1"], outputdata)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

        # 提前停止相关设置
        best_loss = float('inf')
        best_state_dict = None
        patience = self.patience  # 提前停止容忍轮数，建议在 __init__ 中定义 self.pat
        counter = 0

        for epoch in range(self.epochs):
            self.predbase.train()
            running_loss = 0.0
            for inputs1, targets in train_loader:
                self.optimizerbase.zero_grad()
                pred = self.predbase(inputs1)
                loss = self.criterion(pred, targets)
                loss.backward()
                self.optimizerbase.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            if avg_loss < best_loss - 1e-5:  # 加一个很小的阈值避免浮点误差
                best_loss = avg_loss
                best_state_dict = {k: v.clone() for k, v in self.predbase.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
                    break

        # 恢复最佳模型参数
        if best_state_dict is not None:
            self.predbase.load_state_dict(best_state_dict)

    def update(self, transition_dict, choicehz):
        print(f"[PREDM] Update mode: {'Soft' if self.use_target_pred else 'Hard'}")
        states_S1 = torch.tensor(np.array(transition_dict['pred_state_S1']), dtype=torch.float32).to(self.device)
        inputdata = {
            'S1': states_S1,
        }
        outputdata = torch.tensor(np.array(transition_dict['pred_target']), dtype=torch.float32).to(self.device)
        choices = torch.tensor(choicehz).to(self.device)

        dataset = TensorDataset(inputdata["S1"], choices, outputdata)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

        # 提前停止相关设置
        best_loss = float('inf')
        best_state_dict = None
        patience = self.patience  # 提前停止容忍轮数，建议在 __init__ 中定义 self.pat
        counter = 0

        for epoch in range(self.epochs):
            self.pred.train()
            running_loss = 0.0
            for inputs1, inputs3, targets in train_loader:
                self.optimizer.zero_grad()
                pred, pred_k = self.pred(inputs1, inputs3)
                loss = self.criterion(pred, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            if avg_loss < best_loss - 1e-5:  # 加一个很小的阈值避免浮点误差
                best_loss = avg_loss
                best_state_dict = {k: v.clone() for k, v in self.pred.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
                    break

        # 恢复最佳模型参数
        if best_state_dict is not None:
            self.pred.load_state_dict(best_state_dict)

        if self.use_target_pred:
            self.soft_update()
        else:
            self.hard_update()

    def select_index2(self, choices, chazhihz):
        choices = choices.squeeze(1)
        all_selected_indices = []
        buchong_train = []
        for cls in range(self.num_states):
            all_selected_indices1 = []
            all_selected_indices2 = []
            cls_mask = choices[:, cls] == 1
            cls_indices = torch.nonzero(cls_mask, as_tuple=False).view(-1)  # [M]，第cls类的样本索引
            if cls_indices.numel() == 0:
                buchong_train.append(cls)
                continue  # 当前类无样本，跳过
            else:
                cls_chazhi = chazhihz[cls_indices]
                positive_mask = cls_chazhi > 0
                positive_indices_local = torch.nonzero(positive_mask, as_tuple=False).view(-1)

            if positive_indices_local.numel() == 0:
                half_num = max(1, cls_indices.numel() // 4)  # 至少保留1个
                _, sorted_indices_local = torch.topk(cls_chazhi, k=half_num)
                # print("sorted_indices_local", sorted_indices_local)
                # print()
                positive_indices = cls_indices[sorted_indices_local]
            else:
                positive_indices = cls_indices[positive_indices_local]

            min_seq_len1 = self.param[0]
            min_seq_len2 = self.param[1]
            sorted_indices = torch.sort(positive_indices).values
            current_group = [sorted_indices[0].item()]
            for i in range(1, sorted_indices.shape[0]):
                curr_idx = sorted_indices[i].item()
                prev_idx = sorted_indices[i - 1].item()
                if curr_idx == prev_idx + 1:
                    current_group.append(curr_idx)
                else:
                    if len(current_group) >= min_seq_len1:
                        all_selected_indices1.append(torch.tensor(current_group, device=choices.device))
                    elif len(current_group) >= min_seq_len2:
                        all_selected_indices2.append(torch.tensor(current_group, device=choices.device))

                    current_group = [curr_idx]

            # 处理最后一段
            if len(current_group) >= min_seq_len1:
                all_selected_indices1.append(torch.tensor(current_group, device=choices.device))
            elif len(current_group) >= min_seq_len2:
                all_selected_indices2.append(torch.tensor(current_group, device=choices.device))

            if len(all_selected_indices1) > 0:
                all_selected_indices.append(torch.cat(all_selected_indices1, dim=0))
            else:
                if len(all_selected_indices2) > 0:
                    all_selected_indices.append(torch.cat(all_selected_indices2, dim=0))
                # else:
                #     buchong_train.append(cls)
        # 合并所有类的索引
        final_selected_indices = torch.cat(all_selected_indices, dim=0) if all_selected_indices else torch.tensor([],
                                                                                                                  dtype=torch.long,
                                                                                                                  device=choices.device)
        print("si2_train_len", len(final_selected_indices))
        buchong_train = np.unique(buchong_train)
        return final_selected_indices, buchong_train

    def update2(self, transition_dict, chazhihz, choicehz, errorhz):
        print(f"[PREDM] Update mode: {'Soft' if self.use_target_pred else 'Hard'}")
        states_S1 = torch.tensor(np.array(transition_dict['pred_state_S1']), dtype=torch.float32).to(self.device)
        inputdata = {
            'S1': states_S1,
        }
        outputdata = torch.tensor(np.array(transition_dict['pred_target']), dtype=torch.float32).to(self.device)
        choices = torch.tensor(choicehz).to(self.device)
        chazhihz = torch.tensor(chazhihz).to(self.device)

        selectindex, buchong_train = self.select_index2(choices, chazhihz)
        if len(selectindex) != 0:
            dataset = TensorDataset(inputdata["S1"][selectindex], choices[selectindex],
                                    outputdata[selectindex])
            train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

            # 提前停止相关设置
            best_loss = float('inf')
            best_state_dict = None
            patience = self.patience  # 提前停止容忍轮数，建议在 __init__ 中定义 self.pat
            counter = 0

            for epoch in range(self.epochs):
                self.pred.train()
                running_loss = 0.0
                for inputs1, inputs3, targets in train_loader:
                    self.optimizer.zero_grad()
                    pred, pred_k = self.pred(inputs1, inputs3)
                    loss = self.criterion(pred, targets)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)

                if avg_loss < best_loss - 1e-5:  # 加一个很小的阈值避免浮点误差
                    best_loss = avg_loss
                    best_state_dict = {k: v.clone() for k, v in self.pred.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
                        break

            # 恢复最佳模型参数
            if best_state_dict is not None:
                self.pred.load_state_dict(best_state_dict)

            if self.use_target_pred:
                self.soft_update()
            else:
                self.hard_update()

    def soft_update(self):
        for target_param, param in zip(self.target_pred.parameters(), self.pred.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        self.target_pred.load_state_dict(self.pred.state_dict())

    def soft_update2(self,target):
        for pred_param, param in zip(self.pred.parameters(), target.pred.parameters()):
            pred_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * pred_param.data)
        for target_param, param in zip(self.target_pred.parameters(), target.target_pred.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict(self, state, prob):
        state = {
            'S1': torch.tensor(state['S1'], dtype=torch.float32).to(self.device).unsqueeze(0),
        }
        probs = torch.tensor(prob, dtype=torch.float32).to(self.device).unsqueeze(0)
        self.target_pred.eval()
        with torch.no_grad():
            pred, pred_k = self.target_pred(state["S1"], probs)
        return pred.cpu().numpy().flatten(), pred_k.cpu().numpy().flatten()

    def predictbase(self, state):
        state = {
            'S1': torch.tensor(state['S1'], dtype=torch.float32).to(self.device).unsqueeze(0),
        }
        self.predbase.eval()
        with torch.no_grad():
            pred = self.predbase(state["S1"])
        return pred.cpu().numpy().flatten()

    def save(self, directory, name):
        torch.save(self.target_pred.state_dict(), f'{directory}/{name}_pred.pth')
        torch.save(self.predbase.state_dict(), f'{directory}/{name}_predbase.pth')

    def load(self, directory, name):
        self.pred.load_state_dict(
            torch.load(f'{directory}/{name}_pred.pth', map_location=lambda storage, loc: storage))
        self.target_pred.load_state_dict(self.pred.state_dict())

        self.predbase.load_state_dict(
            torch.load(f'{directory}/{name}_predbase.pth', map_location=lambda storage, loc: storage))
