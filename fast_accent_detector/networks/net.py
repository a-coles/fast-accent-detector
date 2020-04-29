'''
Trainable neural network class
'''

import os
import torch

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt


class Net():
    def __init__(self, cfg, device, name=None):
        self.device = device
        self.name = name
        self.cfg = cfg
        self.model = None
        self.train_metrics, self.valid_metrics, self.test_metrics = [], [], []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_metrics(self, log_dir, graph=True):
        metric_names = self.train_metrics[0].keys()
        num_epochs = len(self.train_metrics)
        for metric in metric_names:
            train_vals = [x[metric] for x in self.train_metrics]
            valid_vals = [x[metric] for x in self.valid_metrics]
            header = 'epoch,train_{0},valid_{0}'.format(metric)
            with open(os.path.join(log_dir, '{0}_{1}.csv'.format(self.name, metric)), 'w') as fp:
                fp.write('{0}\n'.format(header))
                for e in range(num_epochs):
                    fp.write('{0},{1},{2}\n'.format(e, train_vals[e], valid_vals[e]))
            if graph:
                plt.plot(list(range(num_epochs)), train_vals, color='blue', label='Train')
                plt.plot(list(range(num_epochs)), valid_vals, color='red', label='Valid')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.legend()
                plt.savefig(os.path.join(log_dir, '{0}_{1}.png'.format(self.name, metric)))
                plt.clf()

    def print_metrics(self, metrics, split='train'):
        to_print = []
        for k, v in metrics.items():
            to_print.append('{0}_{1} {2}'.format(split, k, v))
        to_print = '\t'.join(to_print)
        print(to_print)

    def train(self, train_loader, valid_loader, best_metric='f1', patience=10):
        max_metric = 0.0
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        for epoch in range(self.cfg['num_epochs']):
            print('EPOCH {} of {}'.format(epoch, self.cfg['num_epochs']))
            train_metrics = self.train_epoch(train_loader, opt)
            self.print_metrics(train_metrics, split='train')
            valid_metrics = self.valid_epoch(valid_loader)
            self.print_metrics(valid_metrics, split='valid')

            # Early stop
            valid_losses = [x['loss'] for x in self.valid_metrics]
            last_val_losses = valid_losses[-patience:]
            if epoch > patience:
                stop = True
                for l in last_val_losses:
                    if valid_metrics['loss'] < l:
                        stop = False
                        break
                if stop:
                    print('Early stopping: validation loss has not improved in {0} epochs.'.format(patience))
                    break

            # Log metrics
            self.train_metrics.append(train_metrics)
            self.valid_metrics.append(valid_metrics)
            self.log_metrics(os.path.join('..', 'results'))

            if valid_metrics[best_metric] > max_metric:
                max_metric = valid_metrics[best_metric]
                self.save_model(os.path.join('..', 'models', '{0}.pt'.format(self.name)))
                print('Best {0} so far, saving.'.format(best_metric))
            # self.save_model(os.path.join('..', 'models', '{0}_{1}.pt'.format(self.name, epoch)))

    def train_epoch(self):
        pass

    def valid_epoch(self):
        pass

    def test(self, test_loader):
        test_metrics = self.valid_epoch(test_loader, test=True)
        self.print_metrics(test_metrics, split='test')
        return test_metrics

    def f1_score(self, y_true, y_pred, eps=1e-9):
        y_pred = y_pred.float()
        y_true = y_true.float()
        tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0

        for (t, p) in zip(y_true, y_pred):
            p = 1.0 if p[0] == 0.0 else 0.0
            t = 1.0 if t[0] == 0.0 else 0.0
            if t == 1.0 and p == 1.0:
                tp += 1  # True positive
            elif t == 0.0 and p == 1.0:
                fp += 1  # False positive
            elif t == 1.0 and p == 0.0:
                fn += 1  # False negative
            elif t == 0.0 and p == 0.0:
                tn += 1  # True negative

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return f1
