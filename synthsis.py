
import argparse
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write

from eval import synthesis, get_DNN
from melgan.generator import Generator
from melgan.utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def melgan_synthsis(mel, melgan_checkpoint_path, out_path, config='/opt/tiger/melgan/config/default.yaml'):
    checkpoint = torch.load(melgan_checkpoint_path)
    if config is not None:
        hp = HParam(config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    audio = model.inference(mel)
    audio = audio.cpu().detach().numpy()
    write(out_path, hp.audio.sampling_rate, audio)


def acoustic_synthsis(step, alpha, text):
    print("use griffin-lim and melgan")
    model = get_DNN(step)
    phn = text.text_to_sequence(text, hp.text_cleaners)
    mel, mel_cuda = synthesis(model, phn, alpha)
    mel = torch.Tensor(mel)
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    mel = mel.cuda()
    return mel


def do_synthsis(text, out_path):
    step = 120000
    alpha = 1.0
    config = '/opt/tiger/melgan/config/default.py'
    melgan_checkpoint_path = '/opt/tiger/melgan/chkpt/log/log_aca5990_2025.pt'

    mel = acoustic_synthsis(step, alpha, text)
    melgan_synthsis(mel, melgan_checkpoint_path, out_path, config)