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
import cv2
import os
from collections import Counter
import skimage

import time

from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter

import modules


from modules import path2hash


hoge=path2hash("./data/JKspeech-v_1_0/JKspeech/E01.wav")