'''
File containing functions for preprocessing of data.
'''
from functools import reduce
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks

import argparse
import os


def remove_silence(audio):
    # Lightly adapted from:
    # https://stackoverflow.com/questions/23730796/using-pydub-to-chop-up-a-long-audio-file
    # We consider it silent if quieter than -16 dBFS for at least half a second.
    # (Might not use this if we want to match up with time stamps from transcriptions.)
    # Also it doesn't work right now anyway - debug later.
    audio_parts = split_on_silence(audio, min_silence_len=500, silence_thresh=-16)
    audio = reduce(lambda a, b: a + b, audio_parts)  # Re-combine
    return audio


def stereo_to_mono(audio):
    mono_audio = audio.set_channels(1)
    return mono_audio


def get_speaker_dict(input_dir, corpus):
    # Group together the files belonging to each speaker
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    speaker_files = {}
    for file in files:
        if corpus == 'librispeech':
            speaker_id = file.split('-')[0]
        elif corpus == 'librit':
            speaker_id = file.split('_')[0]
        if speaker_id in speaker_files:
            speaker_files[speaker_id].append(file)
        else:
            speaker_files[speaker_id] = [file]
    return speaker_files


def downsample(audio, sample_rate):
    # Downsamples an audio object to a given rate
    audio = audio.set_frame_rate(sample_rate)
    return audio


def consolidate_speakers(input_dir, speaker_id, speaker_dict):
    # Concatenates all files from specific speaker into one audio object.
    file_list = speaker_dict[speaker_id]
    consolidated = AudioSegment.empty()
    for i, file in enumerate(file_list):
        print(' Processing file {} of {}...'.format(i, len(file_list)))
        file_audio = AudioSegment.from_wav(os.path.join(input_dir, file))
        consolidated += file_audio
    return consolidated


def preprocess(input_dir, output_dir, length_in_sec, corpus):
    speaker_dict = get_speaker_dict(input_dir, corpus)
    speaker_ids = list(speaker_dict.keys())
    for j, speaker_id in enumerate(speaker_ids):
        print('Making files for speaker {} ({} of {})...'.format(speaker_id, j, len(speaker_ids)))
        # Get the concatenated audio for each speaker
        speaker_audio = consolidate_speakers(input_dir, speaker_id, speaker_dict)
        # Remove silence
        # speaker_audio = remove_silence(speaker_audio)
        # Downsample to mono
        speaker_audio = stereo_to_mono(speaker_audio)
        # Librit needs to be downsampled to 16k
        if corpus == 'librit':
            speaker_audio = downsample(speaker_audio, 16000)
        # Split into clips of length
        length = length_in_sec * 1000
        audio_chunks = make_chunks(speaker_audio, length)
        # Export the clips
        for i, chunk in enumerate(audio_chunks):
            chunk.export(os.path.join(output_dir, '{}_{}.wav'.format(speaker_id, i)), format='wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the corpus data.')
    parser.add_argument('corpus', help='The name of the corpus to preprocess. Can be librit or librispeech.')
    parser.add_argument('input_dir', help='The directory containing the audio to preprocess.')
    parser.add_argument('output_dir', help='The directory where the preprocessed audio should go.')
    parser.add_argument('clip_length', help='The length of clip you want, in seconds.', type=int)
    args = parser.parse_args()

    # Make sure corpus name is okay
    valid_corpora = ['librispeech', 'librit']
    if args.corpus not in valid_corpora:
        raise ValueError('Invalid corpus name. Can be one of: {}'.format(valid_corpora))

    # Do the preprocessing
    preprocess(args.input_dir, args.output_dir, args.clip_length, args.corpus)
