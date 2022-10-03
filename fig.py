
import sys
import modules
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
import os

from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter

def save_all_fig(dir_path):
    for filename in os.listdir(dir_path):
        base,ext=os.path.splitext(filename)
        path=f"{dir_path}/{filename}"
        if ext=='.wav':
            data,samplerate=modules.wav_read(path)

            fft_array,fs=modules.path2sgram(path)
            

            fig=plt.figure()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax1=fig.add_subplot(111)
            ax1.axis("off")
            im=ax1.imshow(
            fft_array,
            aspect='auto',
            cmap='jet'
            )

            # cbar=fig.colorbar(im)
            # cbar.set_label('SPL[dBA]')
            
            # 軸設定する。
            # ax1.set_xlabel('Time [s]')
            # ax1.set_ylabel('Frequency [Hz]')

            # # スケールの設定をする。
            # ax1.set_xticks(np.arange(0, 50, 1))
            # ax1.set_yticks(np.arange(0, 20000, 500))
            # ax1.set_xlim(0, 5)
            # ax1.set_ylim(0, 2000)

            plt.savefig(f"./imgs/melspec/{filename}.png")


if __name__=="__main__":
    save_all_fig("./data/JKspeech-v_1_0/JKspeech")