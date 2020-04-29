'''
RL model
'''

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import Net

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt


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
            # print('Example {0}'.format(i))
            # if i > 100:
            #     break
            # if i % 100 == 0:
            #     print('Example {}'.format(i))
            opt.zero_grad()
            # print('true tag:', y)
            x, y = x.to(self.device).float(), y.to(self.device).float()
            hidden = self.model.init_hidden(self.cfg['train_bsz'])

            # Pass input through model
            tag_dist, action_probs, baselines, num_t = self.model(x, hidden, is_training=True)
            # print('action probs', action_probs)
            # print('tag dist', tag_dist)
            # tag = torch.argmax(tag_dist).unsqueeze(0).int()
            tag = F.one_hot(torch.argmax(tag_dist, dim=1)).float()
            # print('true tag:', y)
            # print('pred tag:', tag)

            # Calculate reward/return (all 0s until last timestep)
            # num_t = len(actions)
            # ret = self.gamma * self.model.ActionSelector.get_reward(tag.detach(), y.detach(), num_t)
            final_reward = self.model.ActionSelector.get_reward(tag, y, num_t, self.cfg['train_bsz']).unsqueeze(0)#.permute(1, 0)
            # print('final reward', final_reward.size())
            final_reward = final_reward.permute(1, 0)
            prev_reward = torch.zeros([self.cfg['train_bsz'], num_t])
            # print('prev  reward', prev_reward)
            # rewards = torch.cat((torch.zeros([self.train_bsz, num_t]), final_reward), dim=1)
            rewards = torch.cat((prev_reward,
                                 final_reward), dim=1)
            # rewards = rewards.permute(2, 1, 0)
            # print('rewards', rewards, rewards.size())
            returns = self.get_returns(rewards)
            # print('returns', returns.size())
            # print('ret', ret)
            # rets = torch.concat((torch.zeros([num_t - 1]), ret))
            # print('action probs', action_probs)
            # loss_agent = ret * (-1 * torch.sum(action_probs))
            # print('returns', returns.size())
            # print('baselines', baselines.size())
            ret_hat = returns - baselines
            ret_hat = (ret_hat - ret_hat.mean().expand_as(ret_hat)) / (ret_hat.std(unbiased=False) + 1e-9).expand_as(ret_hat)
            # loss_agent = ret_hat * (-1 * action_probs[-1])
            loss_agent = self.agent_loss(action_probs, ret_hat)
            # print('loss agent', loss_agent)

            loss_classifier = bce(tag_dist, y)
            # print('loss classifier', loss_classifier)
            # loss_agent = self.rl_loss()
            loss_baseline = self.baseline_loss(ret_hat, baselines)
            # print('loss baseline', loss_baseline)
            loss = (-loss_agent) + (loss_classifier) + loss_baseline

            # print(i, 'final reward', final_reward.item(), '\t time', num_t, '\t loss', loss.item())

            loss.backward()
            opt.step()
            loss_epoch += loss.item()
            num_t_epoch += num_t

            # pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, tag)
            # break
        # print('     Epoch on avg {0} timesteps.'.format(num_t_epoch / len(train_loader)))
        # return loss_epoch / len(train_loader), f1_epoch / len(train_loader), num_t_epoch / len(train_loader)
        # return loss_epoch, f1_epoch
        # epoch_stats = np.array([loss_epoch, f1_epoch, num_t_epoch]) / len(train_loader)
        # return epoch_stats
        metrics = {'loss': loss_epoch / len(train_loader),
                   'f1': f1_epoch / len(train_loader),
                   'timesteps': num_t_epoch / len(train_loader)}
        return metrics

    def get_returns(self, rewards):
        returns, R = torch.zeros_like(rewards).to('cuda'), 0
        # print(rewards.size())
        for i, r in enumerate(torch.flip(rewards, dims=(0,))):
            R = r + self.gamma * R
            # returns.insert(0, R)
            returns[i] = R
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        returns = torch.flip(returns, dims=(0,))
        # print('returns', returns)
        # print('ret mean', returns.mean())
        # print('ret std', returns.std(unbiased=False))
        # if returns.size()[0] != 1:
        returns = (returns - returns.mean().expand_as(returns)) / (returns.std(unbiased=False) + 1e-9).expand_as(returns)
        return returns

    def agent_loss(self, action_probs, ret_hat):
        # print('action probs', action_probs.size())
        # print('ret hat', ret_hat.size())
        loss_sum = 0.0
        # for i, r in enumerate(ret_hat):
        #     # print('r', r)
        #     # print('action prob', action_probs[i])
        #     if r < 0:
        #         loss_sum += r * (-1 * torch.log(1 - torch.exp(action_probs[i])))
        #     else:
        #         loss_sum += r * (-1 * action_probs[i])
        # loss_sum = torch.dot(ret_hat, -action_probs)
        # Loop over batch
        for b in ret_hat:
            for i, r in enumerate(b):
                # for i, r in enumerate(t):
                loss_sum += r * (-1 * action_probs[i])
        return loss_sum / self.cfg['train_bsz']

    def baseline_loss(self, ret_hat, baselines):
        # loss_sum = 0.0
        # # for t in range(num_t):
        # #     if t == num_t:
        # #         loss_sum += baselines[t]**2
        # #     else:
        # #         loss_sum += (ret - baseline)**2
        # loss_sum = torch.sum(ret_hat)**2
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
            # print('action probs', action_probs)
            # print('tag dist', tag_dist)
            # tag = torch.argmax(tag_dist).unsqueeze(0).int()
            tag = F.one_hot(torch.argmax(tag_dist, dim=1)).float()
            # print('true tag:', y)
            # print('pred tag:', tag)

            # Calculate reward/return (all 0s until last timestep)
            # num_t = len(actions)
            # ret = self.gamma * self.model.ActionSelector.get_reward(tag.detach(), y.detach(), num_t)
            final_reward = self.model.ActionSelector.get_reward(tag, y, num_t, self.cfg['train_bsz']).unsqueeze(0)#.permute(1, 0)
            # print('final reward', final_reward.size())
            final_reward = final_reward.permute(1, 0)
            prev_reward = torch.zeros([self.cfg['train_bsz'], num_t])
            # print('prev  reward', prev_reward)
            # rewards = torch.cat((torch.zeros([self.train_bsz, num_t]), final_reward), dim=1)
            rewards = torch.cat((prev_reward,
                                 final_reward), dim=1)
            # rewards = rewards.permute(2, 1, 0)
            # print('rewards', rewards, rewards.size())
            returns = self.get_returns(rewards)
            # print('returns', returns.size())
            # print('ret', ret)
            # rets = torch.concat((torch.zeros([num_t - 1]), ret))
            # print('action probs', action_probs)
            # loss_agent = ret * (-1 * torch.sum(action_probs))
            # print('returns', returns.size())
            # print('baselines', baselines.size())
            ret_hat = returns - baselines
            ret_hat = (ret_hat - ret_hat.mean().expand_as(ret_hat)) / (ret_hat.std(unbiased=False) + 1e-9).expand_as(ret_hat)
            # loss_agent = ret_hat * (-1 * action_probs[-1])
            loss_agent = self.agent_loss(action_probs, ret_hat)
            # print('loss agent', loss_agent)

            loss_classifier = bce(tag_dist, y)
            # print('loss classifier', loss_classifier)
            # loss_agent = self.rl_loss()
            loss_baseline = self.baseline_loss(ret_hat, baselines)
            # print('loss baseline', loss_baseline)
            loss = (-loss_agent) + (loss_classifier) + loss_baseline

            # print(i, 'final reward', final_reward.item(), '\t time', num_t, '\t loss', loss.item())

            # loss.backward()
            # opt.step()
            loss_epoch += loss.item()
            num_t_epoch += num_t

            # pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, tag)
            # break
        # print('     Epoch on avg {0} timesteps.'.format(num_t_epoch / len(valid_loader)))
        # return loss_epoch / len(valid_loader), f1_epoch / len(valid_loader)
        # epoch_stats = np.array([loss_epoch, f1_epoch, num_t_epoch]) / len(valid_loader)
        # return epoch_stats
        metrics = {'loss': loss_epoch / len(valid_loader),
                   'f1': f1_epoch / len(valid_loader),
                   'timesteps': num_t_epoch / len(valid_loader)}
        return metrics


class RLModel(nn.Module):
    # Helper class to facilitate keeping all weights in one model
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, device='cuda'):
        super(RLModel, self).__init__()
        # self.LSTM = LSTM()
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
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
            # print('x', x.size())
            # x_t = x[:, t, :].unsqueeze(1)
            x_t = x[:, t, :]
            # print('x_t', x_t.size())
            # print('hidden', hidden)
            # print('hidden[0]', hidden[0].size())
            # print('hidden[1]', hidden[1].size())
            # x_t, hidden = self.lstm(x_t, hidden)
            hidden, cell = self.lstm(x_t, (hidden, cell))
            # print('hidden', hidden.size())
            hiddens.append(hidden)
            # print(torch.stack(hiddens).size())
            hiddens_avg = torch.mean(torch.stack(hiddens), dim=0)
            # print('hiddens avg', hiddens_avg.size())
            # print('hidden[0] after lstm', hidden[0].size())
            baseline = self.Baseline(hiddens_avg.detach())
            # print('baseline', baseline.size())
            baselines.append(baseline.squeeze())
            # print('x_t after lstm', x_t.size())
            # Use RL agent to determine whether to pass to classification or not.
            # Action 0: terminate; action 1: wait
            # print(torch.mean(hiddens_avg, dim=0).size())
            action_dist = self.ActionSelector(torch.mean(hiddens_avg, dim=0))
            # print('action dist', action_dist)
            # action = np.random.choice(self.actions, p=action_dist)
            m = torch.distributions.Categorical(action_dist)
            if is_training:
                action = m.sample()
            else:
                action = torch.argmax(action_dist)
            # print('action', action)
            # # action = torch.argmax(action_dist)
            logprob = m.log_prob(action)
            # print('logprob', logprob)
            action_probs.append(logprob)
            if action.item() == 0:
                decision_made = True
                # Pass to classification
                tag_dist = self.Classifier(hiddens_avg)
                # print('Classifying at time {}'.format(t))
                break
            else:
                # Process the next frame
                continue
        # If we've been through all time steps and did not classify, classify now
        if not decision_made:
            # print('We made no decision! Classifying.')
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
        # self.fc1 = nn.Linear(hidden_dim, 32)
        # self.fc2 = nn.Linear(32, num_classes)
        self.fc1 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # print('x', x.size())
        x = x.squeeze()
        # print('x', x.size())
        out = self.fc1(x)
        # print('out', out.size())
        # out = self.fc2(out)
        out = self.softmax(out)
        # print('out', out.size())
        return out


class Baseline(nn.Module):
    # Calculate the baseline value
    def __init__(self, hidden_dim):
        super(Baseline, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.squeeze()
        # print('b in', x.size())
        out = self.fc(x)
        # print('b out', out.size())
        return out


class ActionSelector(nn.Module):
    # Decide whether to listen to the next frame or make a classification
    def __init__(self, hidden_dim, num_actions):
        super(ActionSelector, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.squeeze()
        # print('x in act sel', x.size())
        out = self.fc(x)
        out = self.softmax(out)
        # print('out', out.size())
        return out

    def accuracy_reward(self, pred_y, true_y):
        # NB: This works with bsz = 1
        # true_y, pred_y = true_y.detach(), pred_y.detach()
        # print('true y', true_y)
        # print('pred y', pred_y)
        rewards = torch.zeros([true_y.size()[0]])
        for i, (p, t) in enumerate(zip(pred_y, true_y)):
            # print('p, t', p, t)
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
        # print('true y 0', true_y[0])
        # true_y = 0 if true_y[0] == 1 else 1
        # # print('true y', true_y)
        # if true_y == 1 and pred_y == 1:
        #     reward = 1  # True positive
        # elif true_y == 0 and pred_y == 1:
        #     reward = -1  # False positive
        # elif true_y == 1 and pred_y == 0:
        #     reward = -1  # False negative
        # elif true_y == 0 and pred_y == 0:
        #     reward = 1  # True negative

        # print('accuracy reward', reward)
        return rewards
        # return torch.tensor(rewards).float().to('cuda')

    def latency_reward(self, t):
        r_lat = float(1 / (t + 1))
        # print('latency reward', r_lat)
        return r_lat

    def get_reward(self, pred_y, true_y, t, bsz):
        # All rewards are 0 except at terminal time step
        if t >= 1998:
            # reward = torch.tensor(-1).float().to('cuda')
            reward = -1 * torch.ones([bsz])
        else:
            reward = self.accuracy_reward(pred_y, true_y) + self.latency_reward(t)
        # return torch.tensor(reward).float().to('cuda')
        return reward
