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
import skimage

import librosa.display
import scipy.ndimage as ndi

from scipy.signal.windows.windows import nuttall
import scipy.signal

import hashlib
from operator import itemgetter
from typing import List, Tuple

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)


from collections import Counter


from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter




def wav_read(path):
  wave,fs=sf.read(path)
  return wave,fs


def wav2sgram(path,n_fft=512,win_length=512):
  wave,fs=wav_read(path)
  sgram=np.abs(librosa.stft(wave,n_fft,win_length))
  return sgram,wave,fs

def path2sgram(path):
  sgram,wave,fs=wav2sgram(path)
  spec_db=librosa.amplitude_to_db(sgram,ref=0)

  # 振幅に変換
  amp = np.abs(spec_db)
  #print(amp.shape)  # --> (8001, 1)

  # 正規化
  # N = len(wave)  # データ数
  # amp_normal = amp / (N / 2)
  # amp_normal[0] /= 2

  return amp,fs

# オーバーラップをかける関数
def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate                     # 全データ長
    Fc = Fs / samplerate                            # フレーム周期
    x_ol = Fs * (1 - (overlap / 100))               # オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap / 100))) /
                (Fc * (1 - (overlap / 100))))       # 抽出するフレーム数（平均化に使うデータ個数）

    array = []                                      # 抽出したデータを入れる空配列の定義

    # forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)                          # 切り出し位置をループ毎に更新
        array.append(data[ps:ps + Fs:1])            # 切り出し位置psからフレームサイズ分抽出して配列に追加
        final_time = (ps + Fs)/samplerate           # 切り出したデータの最終時刻
    return array, N_ave, final_time                 # オーバーラップ抽出されたデータ配列とデータ個数、最終時間を戻り値にする

# ハニング窓をかける関数（振幅補正係数計算付き）
def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)                           # ハニング窓作成
    acf = 1 / (sum(han) / Fs)                       # 振幅補正係数(Amplitude Correction Factor)

    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han         # 窓関数をかける

    return data_array, acf

# dB(デシベル）演算する関数
def db(x, dBref):
    y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す

# 聴感補正(A補正)する関数
def aweightings(f):
    if f[0] == 0:
        f[0] = 1
    ra = (np.power(12194, 2) * np.power(f, 4))/\
         ((np.power(f, 2) + np.power(20.6, 2)) *
          np.sqrt((np.power(f, 2) + np.power(107.7, 2)) *
                  (np.power(f, 2) + np.power(737.9, 2))) *
          (np.power(f, 2) + np.power(12194, 2)))
    a = 20 * np.log10(ra) + 2.00
    return a

# 平均化FFTする関数
def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成
    a_scale = aweightings(fft_axis)                # 聴感補正曲線を計算

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        fft_array.append(db
                        (acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2))
                        , 2e-5))

    fft_array = np.array(fft_array) + a_scale      # 型をndarrayに変換しA特性をかける
    fft_mean = np.mean(fft_array, axis=0)          # 全てのFFT波形の平均を計算

    return fft_array, fft_mean, fft_axis

# ピーク検出関数
def detect_peaks(image, filter_size=3, order=0.3):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))

    # 小さいピーク値を排除（最大ピーク値のorder倍以下のピークは排除）
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def sgram2peaks(arr2D,amp_min=10,CONNECTIVITY_MASK=2,PEAK_NEIGHBORHOOD_SIZE=10,plot=False):
    """
    Extract maximum peaks from the spectogram matrix (arr2D).
    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = scipy.ndimage.morphology.generate_binary_structure(2, CONNECTIVITY_MASK)
    
    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = scipy.ndimage.morphology.iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
    
    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    #file(local_max)
    np.savetxt("txt.txt",local_max)
    
    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (arr2D == 0)
    eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    
    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        print("plot")
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter,c="pink",s=5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return freqs_filter, times_filter

# 波形(x, y)からn個のピークを幅wで検出する関数(xは0から始まる仕様）
def findpeaks(dx, y, n, w):
    index_all = list(signal.argrelmax(y, order=w))                  # scipyのピーク検出
    index = []                                                      # ピーク指標の空リスト
    peaks = []                                                      # ピーク値の空リスト

    # n個分のピーク情報(指標、値）を格納
    for i in range(n):
        # n個のピークに満たない場合は途中でループを抜ける（エラー処理）
        if i >= len(index_all[0]):
            break
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])

    # 個数の足りない分を0で埋める（エラー処理）
    if len(index) != n:
        index = index + ([0] * (n - len(index)))
        peaks = peaks + ([0] * (n - len(peaks)))
    
    index = np.array(index) * dx                                  # xの分解能dxをかけて指標を物理軸に変換
    peaks = np.array(peaks)
    return index, peaks

# スペクトログラムからピークを検出する関数
# fft_array=スペクトログラム（転置前）
# dt, df=スペクトログラムの時間分解能, 周波数分解能
# num_peaks=1つの周波数軸で検出するピーク数
# w=ノイズ対策用の幅（order）
# max_peaks=最終的にスペクトログラムから抽出するピーク数（振幅の大きい順）
def findpeaks_2d(fft_array, dt, df, num_peaks, w, max_peaks):

    # ピーク情報を初期化する
    time_index = np.zeros((len(fft_array), num_peaks))
    freq_index = np.zeros((len(fft_array), num_peaks))
    freq_peaks = np.zeros((len(fft_array), num_peaks))

    # 各周波数軸毎にピークを検出する
    for i in range(len(fft_array)):
        index, peaks = findpeaks(df, fft_array[i], n=num_peaks, w=w)    # ピーク検出
        freq_peaks[i] = peaks                                           # 検出したピーク値(振幅)を格納
        freq_index[i] = index                                           # 検出したピーク位置(周波数)を格納
        time_index[i] = np.full(num_peaks, i) * dt                      # 検出したピーク位置(時間)を格納

    # 平坦化する
    freq_peaks = freq_peaks.ravel()
    freq_index = freq_index.ravel()
    time_index = time_index.ravel()

    # ピークの大きい順（降順）にソートする
    freq_peaks_sort = np.sort(freq_peaks)[::-1]
    freq_index_sort = freq_index[np.argsort(freq_peaks)[::-1]]
    time_index_sort = time_index[np.argsort(freq_peaks)[::-1]]



    return freq_index_sort[:max_peaks], freq_peaks_sort[:max_peaks], time_index_sort[:max_peaks]



def pairling_peaks(freq_index,time_index,fanvalue=2,mintdt=0,maxtdt=2,minfdt=-3,maxfdt=3):

  '''
  ピークをペアにする関数
  ------------
  fanvalue n個先までのピークをチェックする
  maxtdt,mintdt 時間の位置差分がnまでにあったらペアリング
  maxfdt,mintdt 周波数のいち差分がnまでにあったらペアリング
  -------------
  pair_landmark f1,f2,dt,t1
  '''

  landmarks=[]
  ntimes=len(freq_index)
  #print(f'fffffffffffffffff{ntimes}')
  for i in range(ntimes):
    t1,f1=time_index[i],freq_index[i]
    for j in range(fanvalue):
      tmp=i+j
      if tmp>=ntimes:
        break
      #print(tmp)
      t2,f2=time_index[tmp],freq_index[tmp]
      dt=t2-t1
      df=f2-f1
      if not mintdt<=dt and dt<=maxtdt and minfdt<=df and df<=maxfdt:
        continue
      landmarks.append([f1,f2,dt,t1])
  
  return landmarks

def peaks2hash(freq_index,time_index,fanvalue=30,mintdt=-2,maxtdt=2,minfdt=-3,maxfdt=3):
  hash_marks={}
  pairs=pairling_peaks(freq_index,time_index,fanvalue,mintdt,maxtdt,minfdt,maxfdt)
  for f1,f2,dt,t1 in pairs:
    info=f'{f1}{f2}{dt}'
    hash=hashlib.sha224(info.encode()).hexdigest()
    hash_marks[hash]=t1
  return hash_marks,pairs

def path2hash(path,graph=False):
  data,samplerate=wav_read(path)
  x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸の作成
  # Fsとoverlapでスペクトログラムの分解能を調整する。
  Fs = 4096                                   # フレームサイズ
  overlap = 90                                # オーバーラップ率

  # オーバーラップ抽出された時間波形配列
  time_array, N_ave, final_time = ov(data, samplerate, Fs, overlap)

  # ハニング窓関数をかける
  time_array, acf = hanning(time_array, Fs, N_ave)

  # FFTをかける
  fft_array, fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf)

  '''
  #ローパスフィルタ
  lowpass=ndimage.gaussian_filter(fft_array,2)
  fft_array-=lowpass
  '''

  # ピーク検出する
  spec_dt = final_time / N_ave
  freq_index, freq_peaks, time_index = findpeaks_2d(fft_array, spec_dt, fft_axis[1], num_peaks=10000, w=10, max_peaks=1000)


  #print(f"freq_peaks==={freq_peaks}")
  #print(f"freq_index==={freq_index}")
  #print(f"time_index===={time_index}")


  # スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
  #fft_array = fft_array.T

  hashmarks,pairs=peaks2hash(freq_index,time_index,fanvalue=10)

  if graph==True:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # データをプロットする。
    im = ax1.imshow(fft_array,
                    vmin=0, vmax=80,
                    extent=[0, final_time, 0, samplerate],
                    aspect='auto',
                    cmap='jet')

    # カラーバーを設定する。
    cbar = fig.colorbar(im)
    cbar.set_label('SPL [dBA]')

    # ピークを描画する。
    for i in range(len(time_index)):
      ax1.scatter(time_index[i], freq_index[i], s=5, facecolor='None', linewidth=3, edgecolors='white')

    for tmp in pairs:
      f1,f2,dt,t1=tmp
      #print(f'{f1}ti iiiiii {t1},,,,,,,{f2}:::::{t1+dt}')
      ax1.plot([t1,t1+dt],[f1,f2],color="white")
      ax1.hist(dt,bins=20)

    # 軸設定する。
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')

    # スケールの設定をする。
    ax1.set_xticks(np.arange(0, 50, 1))
    ax1.set_yticks(np.arange(0, 20000, 500))
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 2000)

    # グラフを表示する。
    plt.show()
    plt.close()

  return hashmarks


def hashmatching(hash1,hash2,graph=False):
  i=0
  ts1=[]
  ts2=[]
  cnt=0
  tmp=[]
  for h in hash2.keys():
    if h in hash1:
      t1,t2=hash1[h],hash2[h]
      #print(f"t1,t2={t1},{t2}")
      ts1.append(t1)
      ts2.append(t2)
      tmp.append([t1,t2])
      i+=1

  diffs={}
  forgraph=[]
  for j in range(len(tmp)):
    t1,t2=tmp[j]
    diff=abs(t2-t1)
    diff=round(diff, 1)
    # print(f"{diff}")
    forgraph.append(diff)
    try:
      diffs[diff]+=1
    except KeyError:
      diffs[diff]=1
  #print(f"tmpは{tmp}")
  diffs_sorted=sorted(diffs.items(),key=lambda x:x[1],reverse=True)  
  n,bins,patches=plt.hist(forgraph,density=True,bins=30)
  plt.close()
  n.sort()

  if graph==True and i>0:
    print("hogeghoegh")
    plt.scatter(ts1,ts2,c="pink",s=5)
    his=list(map(lambda x,y:x-y,ts1,ts2))
    plt.grid(True)
    plt.show()
    plt.close()
    plt.hist(his,density=True,bins=30)
    plt.savefig("./tmp/hist_正規化.png")
    plt.show()
    plt.close()
    plt.hist(forgraph,density=False,bins=30)
    plt.savefig("./tmp/hist2.png")
    plt.show()
    plt.close() 

  #print(f'一致={ts1}')
  #print(cnt)
  #print(len(tmp))
  #print(diffs_sorted)
  #print(f"数は{diffs_sorted}  iは{i}")
  return n[-1]

def img2peaks(path,min_distance=2):
  '''
  画像を特徴点取ってペアにしてハッシュにする関数
  https://scikit-image.org/docs/dev/api/skimage.feature.html#peak-local-max
  ----------
  path : str
    画画のパス(path)
  ---------
  x : list?
  y : list?
    座標リスト
  '''
  x,y=[],[]
  img = skimage.color.rgb2gray(skimage.img_as_float(plt.imread(path)))
  cor=detect_peaks(img)
  for i in range(len(cor)):
    x.append(cor[i][0])
    y.append(cor[i][1])
  return x,y


def path2cqt2sgram(path,hop_length=1024,n_octave=7,bins_per_octave=32,window='hann',fmin=librosa.note_to_hz('C1')):
  '''
  画像を特徴点取ってペアにしてハッシュにする関数
  メモ:sgram2peaksのamp_min=1
  '''
  wav,fs=wav_read(path)
  c=librosa.cqt(wav,sr=fs,
                hop_length=hop_length,
                n_bins=n_octave*bins_per_octave,
                bins_per_octave=bins_per_octave,
                window=window,
                fmin=fmin
                )
  c=abs(c)
  return c