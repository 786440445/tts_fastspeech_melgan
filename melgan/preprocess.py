import os
import glob
import subprocess
import tqdm
import torch
import argparse
import numpy as np

from melgan.utils.stft import TacotronSTFT
from melgan.utils.hparams import HParam


def main(hp, args):
    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)

    # for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
    #     sr, wav = read_wav_np(wavpath)
    #     assert sr == hp.audio.sampling_rate, \
    #         "sample rate mismatch. expected %d, got %d at %s" % \
    #         (hp.audio.sampling_rate, sr, wavpath)
        
    #     if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
    #         wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
    #                 mode='constant', constant_values=0.0)

    #     wav = torch.from_numpy(wav).unsqueeze(0)
    #     mel = stft.mel_spectrogram(wav)
        
    #     melpath = wavpath.replace('.wav', '.mel')
    #     # torch.save(mel, melpath)
    split_train_valid(args.data_path)


def save_wavs(filepaths, save_path):
    for filepath in tqdm.tqdm(filepaths, desc='split train and valid dataset:'):
        melpath = filepath.replace('.wav', '.mel')
        subprocess.Popen(f'cp {filepath} {save_path}', stdout=subprocess.PIPE, shell=True).communicate()
        subprocess.Popen(f'cp {melpath} {save_path}', stdout=subprocess.PIPE, shell=True).communicate()


def split_train_valid(data_dir):
    rate = 0.9
    wavfiles = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    length = len(wavfiles)
    sort_wavs = sorted(wavfiles)
    train_wavs = sort_wavs[:int(rate * length)]
    valid_wavs = sort_wavs[int(rate * length):]
    train_path = os.path.join(data_dir, 'train')
    valid_path = os.path.join(data_dir, 'valid')
    save_wavs(train_wavs, train_path)
    save_wavs(valid_wavs, valid_path)


if __name__ == '__main__':
    # python3 preprocess.py -c ./config/default.yaml -d /opt/tiger/speech_data/LJSpeech-1.1
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)