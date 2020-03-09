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
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            input_file_path = os.path.join(input_dir, filename)
            (rate, sig) = wav.read(input_file_path)
            mfcc_feat = mfcc(sig, rate, nfft=551)
            print("Saving MFCCs for {}".format(filename))
            np.save(os.path.join(output_dir, '{}.txt'.format(os.path.splitext(filename)[0])), mfcc_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from the corpus data.')
    parser.add_argument('input_dir', help='The directory containing the data to extract features from.')
    parser.add_argument('output_dir', help='The directory where the extracted features should go.')
    args = parser.parse_args()

    extract_mfccs(args.input_dir, args.output_dir)
