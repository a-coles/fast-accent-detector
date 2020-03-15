'''
File containing feature extraction methods.
Input should be the preprocessed version of the corpus.
'''

import scipy.io.wavfile as wav
import os
import argparse
import numpy as np

from python_speech_features import mfcc


def extract_mfccs(input_dir, output_dir):
    # Extract mfccs for all wav files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mfcc_feats = []
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            input_file_path = os.path.join(input_dir, filename)
            (rate, sig) = wav.read(input_file_path)
            mfcc_feat = mfcc(sig, rate, nfft=551)
            mfcc_feats.append(mfcc_feat)
    mfcc_feats = np.array(mfcc_feats)

    if args.normalize:
        mean, std = np.mean(mfcc_feats, axis=0), np.std(mfcc_feats, axis=0)
        mfcc_norm = (mfcc_feats - mean) / std

    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            if args.normalize:
                to_save = mfcc_norm[i]
            else:
                to_save = mfcc_feats[i]
            print("Saving MFCCs for {}".format(filename))
            np.save(os.path.join(output_dir, '{}.txt'.format(os.path.splitext(filename)[0])), to_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from the corpus data.')
    parser.add_argument('input_dir', help='The directory containing the data to extract features from.')
    parser.add_argument('output_dir', help='The directory where the extracted features should go.')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize feature-wise.')
    args = parser.parse_args()

    extract_mfccs(args.input_dir, args.output_dir)
