'''
Training routines
'''

import argparse
import json
import os
import torch
import torch.nn as nn

from networks.reinforce import RL
from torch.utils.data import DataLoader
from utils.dataset import AccentDataset, train_test_split

if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Training config.')
    parser.add_argument('--name', type=str, default=None, help='Name for model.')
    parser.add_argument('--continue_model', type=str, help='Path to model for continuing training.')
    args = parser.parse_args()

    # Load config
    # with open(os.path.join('config', 'reinforce.json'), 'r') as fp:
    with open(args.config_file, 'r') as fp:
        cfg = json.load(fp)

    # Create training and valid datasets
    us_dir, uk_dir = os.path.abspath('../datasets/librispeech_mfcc_norm'), os.path.abspath('../datasets/librit_mfcc_norm')
    dataset = AccentDataset(us_dir, uk_dir)
    train_dataset, valid_dataset, _ = train_test_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['train_bsz'], shuffle=True, drop_last=True)

    # Training loop
    # criterion = nn.CrossEntropyLoss()
    model = RL(cfg, device, name=args.name)
    if args.continue_model:
        print('Continuing training from {0}'.format(args.continue_model))
        model.load_model(args.continue_model)
    # model.train(train_loader, valid_loader,
    #             lr=cfg['lr'], train_bsz=cfg['train_bsz'], valid_bsz=cfg['train_bsz'], num_epochs=cfg['num_epochs'])
    model.train(train_loader, valid_loader, best_metric='f1', patience=100)

    # model.log_learning_curves(os.path.join('..', 'results'))
    # model.log_f1(os.path.join('..', 'results'))
    # model.log_f1(os.path.join('..', 'results'))
    # model.save_model(os.path.join('..', 'models', 'baseline.pt'))
