'''
Data loading
'''

import math
import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset


class AccentDataset(Dataset):
    def __init__(self, us_dir, uk_dir, transform=None):
        random.seed(0)
        us_files = [os.path.join(us_dir, f) for f in os.listdir(us_dir) if f.endswith('.npy')]
        us_examples = [(x, 0) for x in us_files]
        uk_files = [os.path.join(uk_dir, f) for f in os.listdir(uk_dir) if f.endswith('.npy')]
        uk_examples = [(x, 1) for x in uk_files]
        self.examples = us_examples + uk_examples
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        x = np.load(example[0])
        y = example[1]
        return (x, y)


def train_test_split(dataset):
    len_train, len_test = math.floor(0.8 * len(dataset)), math.floor(0.1 * len(dataset))
    train, valid, test = torch.utils.data.random_split(dataset, [len_train, len_test + 1, len_test])
    return train, valid, test
