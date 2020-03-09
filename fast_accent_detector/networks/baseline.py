'''
No-RL baseline
'''

import os
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt


class LSTM():
    def __init__(self, cfg, device, name=None):
        self.device = device
        self.name = name
        self.cfg = cfg
        self.train_bsz = cfg['train_bsz']
        self.model = LSTMModel(input_dim=13, hidden_dim=cfg['hidden_dim'], num_classes=2)

        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        self.train_f1, self.valid_f1, self.test_f1 = [], [], []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_learning_curves(self, log_dir, graph=True):
        # Logs the learning curve info to a csv.
        header = 'epoch,train_loss,valid_loss'
        num_epochs = len(self.train_losses)
        with open(os.path.join(log_dir, '{0}_learning_curves.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_losses[e], self.valid_losses[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_losses, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_losses, color='red', label='Valid')
            plt.title('Cross-entropy loss over training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_learning_curves.png'.format(self.name)))
            plt.clf()

    def log_metrics(self, log_dir, graph=True):
        # Logs evaluation metrics (BLEU, etc.) to a csv.
        header = 'epoch,train_f1,valid_f1'
        num_epochs = len(self.train_f1)
        with open(os.path.join(log_dir, '{0}_metrics.csv'.format(self.name)), 'w') as fp:
            fp.write('{0}\n'.format(header))
            for e in range(num_epochs):
                fp.write('{0},{1},{2}\n'.format(e, self.train_f1[e], self.valid_f1[e]))
        if graph:
            plt.plot(list(range(num_epochs)), self.train_f1, color='blue', label='Train')
            plt.plot(list(range(num_epochs)), self.valid_f1, color='red', label='Valid')
            plt.title('F1 score over training')
            plt.xlabel('Epoch')
            plt.ylabel('F1')
            plt.legend()
            plt.savefig(os.path.join(log_dir, '{0}_metrics.png'.format(self.name)))
            plt.clf()

    def f1_score(self, y_true, y_pred, eps=1e-9, beta=1, threshold=2):
        y_pred = y_pred.float()
        y_true = y_true.float()

        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))

        return torch.mean(
            (precision * recall).
            div(precision + recall + eps).
            mul(2)).item()

    def train(self, train_loader, valid_loader, loss_fn=None, lr=1e-2, train_bsz=1, valid_bsz=1, num_epochs=1):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            print('EPOCH {} of {}'.format(epoch, num_epochs))
            train_loss, train_f1 = self.train_epoch(train_loader, opt, loss_fn=loss_fn)
            print('train_loss', train_loss, '\t train_f1', train_f1)
            valid_loss, valid_f1 = self.valid_epoch(valid_loader, loss_fn=loss_fn)
            print('valid_loss', valid_loss, '\t valid_f1', valid_f1)
            self.train_losses.append(train_loss)
            self.train_f1.append(train_f1)
            self.valid_losses.append(valid_loss)
            self.valid_f1.append(valid_f1)

    def train_epoch(self, train_loader, opt, loss_fn=None):
        self.model.train()
        loss_epoch, f1_epoch = 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            if i > 4:  # DEBUG
                break
            opt.zero_grad()
            x, y = x.to(self.device).float(), y.to(self.device).long()
            hidden = self.model.init_hidden(self.train_bsz)
            out = self.model(x, hidden)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_epoch += loss.item()
            pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, pred)
        return loss_epoch / self.train_bsz, f1_epoch / self.train_bsz

    def valid_epoch(self, valid_loader, loss_fn=None):
        self.model.eval()
        loss_epoch, f1_epoch = 0.0, 0.0
        for i, (x, y) in enumerate(valid_loader):
            if i > 4:  # DEBUG
                break
            x, y = x.to(self.device).float(), y.to(self.device).long()
            hidden = self.model.init_hidden(self.train_bsz)
            out = self.model(x, hidden)
            loss = loss_fn(out, y)
            loss_epoch += loss.item()
            pred = torch.argmax(out, axis=1)
            f1_epoch += self.f1_score(y, pred)
        return loss_epoch / self.train_bsz, f1_epoch / self.train_bsz


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, bsz):
        # return torch.zeros(self.num_layers, bsz, self.hidden_dim)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, bsz, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, bsz, self.hidden_dim)
        return (h0, c0)

    def forward(self, x, hidden):
        # print(x.size())
        # print(hidden.size())
        out, hidden = self.lstm(x, hidden)
        # out = out.contiguous().view(-1, self.hidden_dim)  # Stack
        out = self.fc(out)
        return out
