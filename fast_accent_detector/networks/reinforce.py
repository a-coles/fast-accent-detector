'''
RL model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import Net

import matplotlib
matplotlib.use('Agg')  # noqa


class RL(Net):
    def __init__(self, cfg, device, name=None):
        super().__init__(cfg, device, name=name)
        self.gamma = 0.99
        self.model = RLModel(input_dim=13,
                             hidden_dim=cfg['hidden_dim'],
                             num_layers=1,
                             num_classes=2).to(self.device)

    def train_epoch(self, train_loader, opt):
        self.model.train()
        bce = nn.BCELoss()
        loss_epoch, f1_epoch, num_t_epoch = 0.0, 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            x, y = x.to(self.device).float(), y.to(self.device).float()
            hidden = self.model.init_hidden(self.cfg['train_bsz'])

            # Pass input through model
            tag_dist, action_probs, baselines, num_t = self.model(x, hidden, is_training=True)
            tag = F.one_hot(torch.argmax(tag_dist, dim=1)).float()

            # Calculate reward/return (all 0s until last timestep)
            final_reward = self.model.ActionSelector.get_reward(tag, y, num_t, self.cfg['train_bsz']).unsqueeze(0)
            final_reward = final_reward.permute(1, 0)
            prev_reward = torch.zeros([self.cfg['train_bsz'], num_t])
            rewards = torch.cat((prev_reward,
                                 final_reward), dim=1)
            returns = self.get_returns(rewards)
            ret_hat = returns - baselines
            ret_hat = (ret_hat - ret_hat.mean().expand_as(ret_hat)) / (ret_hat.std(unbiased=False) + 1e-9).expand_as(ret_hat)

            # Calculate losses
            loss_agent = self.agent_loss(action_probs, ret_hat)
            loss_classifier = bce(tag_dist, y)
            loss_baseline = self.baseline_loss(ret_hat, baselines)
            loss = (-loss_agent) + (loss_classifier) + loss_baseline

            loss.backward()
            opt.step()
            loss_epoch += loss.item()
            num_t_epoch += num_t
            f1_epoch += self.f1_score(y, tag)

        metrics = {'loss': loss_epoch / len(train_loader),
                   'f1': f1_epoch / len(train_loader),
                   'timesteps': (num_t_epoch / len(train_loader)) + 2}  # Off-by-2 correction for edges
        return metrics

    def get_returns(self, rewards):
        returns, R = torch.zeros_like(rewards).to('cuda'), 0
        for i, r in enumerate(torch.flip(rewards, dims=(0,))):
            R = r + self.gamma * R
            returns[i] = R
        returns = torch.flip(returns, dims=(0,))
        returns = (returns - returns.mean().expand_as(returns)) / (returns.std(unbiased=False) + 1e-9).expand_as(returns)
        return returns

    def agent_loss(self, action_probs, ret_hat):
        loss_sum = 0.0
        # Loop over batch
        for b in ret_hat:
            for i, r in enumerate(b):
                loss_sum += r * (-1 * action_probs[i])
        return loss_sum / self.cfg['train_bsz']

    def baseline_loss(self, ret_hat, baselines):
        mse = nn.MSELoss()
        loss_sum = mse(baselines, ret_hat)
        return loss_sum

    def valid_epoch(self, valid_loader, test=False):
        self.model.eval()
        bce = nn.BCELoss()
        loss_epoch, f1_epoch, num_t_epoch = 0.0, 0.0, 0
        for i, (x, y) in enumerate(valid_loader):
            x, y = x.to(self.device).float(), y.to(self.device).float()
            hidden = self.model.init_hidden(self.cfg['train_bsz'])

            # Pass input through model
            tag_dist, action_probs, baselines, num_t = self.model(x, hidden, is_training=not test)
            tag = F.one_hot(torch.argmax(tag_dist, dim=1)).float()

            # Calculate reward/return (all 0s until last timestep)
            final_reward = self.model.ActionSelector.get_reward(tag, y, num_t, self.cfg['train_bsz']).unsqueeze(0)
            final_reward = final_reward.permute(1, 0)
            prev_reward = torch.zeros([self.cfg['train_bsz'], num_t])
            rewards = torch.cat((prev_reward,
                                 final_reward), dim=1)
            returns = self.get_returns(rewards)
            ret_hat = returns - baselines
            ret_hat = (ret_hat - ret_hat.mean().expand_as(ret_hat)) / (ret_hat.std(unbiased=False) + 1e-9).expand_as(ret_hat)
            loss_agent = self.agent_loss(action_probs, ret_hat)

            # Calculate losses
            loss_classifier = bce(tag_dist, y)
            loss_baseline = self.baseline_loss(ret_hat, baselines)
            loss = (-loss_agent) + (loss_classifier) + loss_baseline

            loss_epoch += loss.item()
            num_t_epoch += num_t
            f1_epoch += self.f1_score(y, tag)

        metrics = {'loss': loss_epoch / len(valid_loader),
                   'f1': f1_epoch / len(valid_loader),
                   'timesteps': (num_t_epoch / len(valid_loader)) + 2}  # Off-by-2 correction for edges
        return metrics


class RLModel(nn.Module):
    # Helper class to facilitate keeping all weights in one model
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, device='cuda'):
        super(RLModel, self).__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.ActionSelector = ActionSelector(hidden_dim=hidden_dim, num_actions=2)
        self.Classifier = Classifier(hidden_dim, num_classes)
        self.Baseline = Baseline(hidden_dim)
        self.actions = [0, 1]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden, is_training=True):
        # Pass audio input through LSTM, one timestep at a time.
        hidden, cell = hidden
        decision_made = False
        action_probs, baselines = [], []
        hiddens = []
        for t in range(x.size()[1]):
            x_t = x[:, t, :]
            hidden, cell = self.lstm(x_t, (hidden, cell))
            hiddens.append(hidden)
            hiddens_avg = torch.mean(torch.stack(hiddens), dim=0)

            baseline = self.Baseline(hiddens_avg.detach())
            baselines.append(baseline.squeeze())

            # Use RL agent to determine whether to pass to classification or not.
            # Action 0: terminate; action 1: wait
            action_dist = self.ActionSelector(torch.mean(hiddens_avg, dim=0))
            m = torch.distributions.Categorical(action_dist)
            if is_training:
                action = m.sample()
            else:
                action = torch.argmax(action_dist)
            logprob = m.log_prob(action)
            action_probs.append(logprob)

            # Take action
            if action.item() == 0:
                decision_made = True
                # Pass to classification
                tag_dist = self.Classifier(hiddens_avg)
                break
            else:
                # Process the next frame
                continue

        # If we've been through all time steps and did not classify, classify now
        if not decision_made:
            tag_dist = self.Classifier(hiddens_avg)
        return tag_dist, torch.stack(action_probs), torch.stack(baselines).permute(1, 0), t

    def init_hidden(self, bsz):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(bsz, self.hidden_dim).to('cuda')
        c0 = torch.zeros(bsz, self.hidden_dim).to('cuda')
        return (h0, c0)


class Classifier(nn.Module):
    # Decide whether to listen to the next frame or make a classification
    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.squeeze()
        out = self.fc1(x)
        out = self.softmax(out)
        return out


class Baseline(nn.Module):
    # Calculate the baseline value
    def __init__(self, hidden_dim):
        super(Baseline, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.squeeze()
        out = self.fc(x)
        return out


class ActionSelector(nn.Module):
    # Decide whether to listen to the next frame or make a classification
    def __init__(self, hidden_dim, num_actions):
        super(ActionSelector, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.squeeze()
        out = self.fc(x)
        out = self.softmax(out)
        return out

    def accuracy_reward(self, pred_y, true_y):
        rewards = torch.zeros([true_y.size()[0]])
        for i, (p, t) in enumerate(zip(pred_y, true_y)):
            p = 1.0 if p[0] == 0.0 else 0.0
            t = 1.0 if t[0] == 0.0 else 0.0
            if t == 1.0 and p == 1.0:
                reward = 1.0  # True positive
            elif t == 0.0 and p == 1.0:
                reward = -1.0  # False positive
            elif t == 1.0 and p == 0.0:
                reward = -1.0  # False negative
            elif t == 0.0 and p == 0.0:
                reward = 1.0  # True negative
            rewards[i] = reward
        return rewards

    def latency_reward(self, t):
        r_lat = float(1 / (t + 1))
        return r_lat

    def get_reward(self, pred_y, true_y, t, bsz):
        # All rewards are 0 except at terminal time step
        if t >= 1998:
            reward = -1 * torch.ones([bsz])
        else:
            reward = self.accuracy_reward(pred_y, true_y) + self.latency_reward(t)
        return reward
