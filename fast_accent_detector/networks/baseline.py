'''
No-RL baseline
'''

import torch
import torch.nn as nn

from .net import Net


class LSTM(Net):
    def __init__(self, cfg, device, name=None):
        super().__init__(cfg, device, name=name)
        self.model = LSTMModel(input_dim=13,
                               hidden_dim=cfg['hidden_dim'],
                               num_classes=2,
                               num_layers=1,
                               device=device).to(device)

    def train_epoch(self, train_loader, opt):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        loss_epoch, f1_epoch = 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            x, y = x.to(self.device).float(), y.to(self.device).long()
            hidden = self.model.init_hidden(self.cfg['train_bsz'])
            out = self.model(x, hidden)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_epoch += loss.item()
            pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, pred)
        metrics = {'loss': loss_epoch / len(train_loader),
                   'f1': f1_epoch / len(train_loader)}
        return metrics

    def valid_epoch(self, valid_loader, test=False):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        loss_epoch, f1_epoch = 0.0, 0.0
        for i, (x, y) in enumerate(valid_loader):
            x, y = x.to(self.device).float(), y.to(self.device).long()
            hidden = self.model.init_hidden(self.cfg['train_bsz'])
            out = self.model(x, hidden)
            loss = loss_fn(out, y)
            loss_epoch += loss.item()
            pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, pred)
        metrics = {'loss': loss_epoch / len(valid_loader),
                   'f1': f1_epoch / len(valid_loader)}
        return metrics


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, device='cuda'):
        super(LSTMModel, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, bsz):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, bsz, self.hidden_dim).to('cuda')
        c0 = torch.zeros(self.num_layers, bsz, self.hidden_dim).to('cuda')
        return (h0, c0)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out
