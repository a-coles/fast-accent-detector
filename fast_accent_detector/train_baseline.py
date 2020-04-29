'''
Training routines
'''

import argparse
import json
import os
import torch
# import torch.nn as nn

from networks.baseline import LSTM
from torch.utils.data import DataLoader
from utils.dataset import AccentDataset, train_test_split

if __name__ == '__main__':
    torch.manual_seed(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Load config
    with open(os.path.join('config', 'baseline2.json'), 'r') as fp:
        cfg = json.load(fp)

    # Create training and valid datasets
    us_dir, uk_dir = os.path.abspath('../datasets/librispeech_mfcc_norm'), os.path.abspath('../datasets/librit_mfcc_norm')
    dataset = AccentDataset(us_dir, uk_dir)
    train_dataset, valid_dataset, _ = train_test_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)

    # Training loop
    # criterion = nn.CrossEntropyLoss()
    model = LSTM(cfg, device, name='baseline64_norm')
    model.train(train_loader, valid_loader, best_metric='f1', patience=10)
