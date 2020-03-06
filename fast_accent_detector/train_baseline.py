'''
Training routines
'''

import argparse
import os
import torch

from torch.utils.data import DataLoader
from utils.dataset import AccentDataset, train_test_split

if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Create training and valid datasets
    us_dir, uk_dir = os.path.abspath('../datasets/librispeech_mfcc_np'), os.path.abspath('../datasets/librit_mfcc_np')
    dataset = AccentDataset(us_dir, uk_dir)
    train_dataset, valid_dataset, _ = train_test_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    for i, (x, y) in enumerate(train_loader):
        print(x)
        print(y)
        break
