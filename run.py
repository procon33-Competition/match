import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.ndimage as ndi
import soundfile as sf
import librosa
import math
import pywt
import librosa
import glob
import time
import pickle
import hashlib
from collections import Counter

from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter

import modules



def run(path,nspeech,funcs):
    '''
    本番時に動かす関数
    選択したwavをpathに突っ込む

    param
    -----------------------
    path : str
        wavのパス

    nspeech : int
        今回の問題の読みデータ数
    
    funcs : funcution
        分析する関数
        可変長引数で取る
    '''
    wav,fs=modules.wav_read(path)
    for func in funcs:
        tmp=func(path)
    return tmp

if __name__=="__main__":
    run("data/sample_Q_202205/sample_Q_202205/sample_Q_E01/problem1.wav",3,modules.path2hash)