'''
Testing routines
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
    parser.add_argument('model', type=str, help='Path to model to test.')
    parser.add_argument('cfg', type=str, help='Path to config file.')
    parser.add_argument('--name', type=str, help='Name for model to test.')
    args = parser.parse_args()

    # Load config
    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)

    # Create test datasets
    us_dir, uk_dir = os.path.abspath('../datasets/librispeech_mfcc_norm'), os.path.abspath('../datasets/librit_mfcc_norm')
    dataset = AccentDataset(us_dir, uk_dir)
    _, _, test_dataset = train_test_split(dataset)
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_bsz'], shuffle=True, drop_last=True)

    # Load and test model
    model = LSTM(cfg, device, name=args.name)
    model.load_model(args.model)
    model.test(test_loader)
