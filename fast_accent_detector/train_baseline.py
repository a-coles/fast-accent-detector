'''
Training routines for baseline
'''

import argparse
import json
import os
import torch

from networks.baseline import LSTM
from torch.utils.data import DataLoader
from utils.dataset import AccentDataset, train_test_split

if __name__ == '__main__':
    torch.manual_seed(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    parser.add_argument('--name', type=str, default=None, help='Name for model.')
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)

    # Create training and valid datasets
    us_dir, uk_dir = os.path.abspath('../datasets/librispeech_mfcc_norm'), os.path.abspath('../datasets/librit_mfcc_norm')
    dataset = AccentDataset(us_dir, uk_dir)
    train_dataset, valid_dataset, _ = train_test_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)

    # Training loop
    model = LSTM(cfg, device, name=args.name)
    model.train(train_loader, valid_loader, best_metric='f1', patience=10)
