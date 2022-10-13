import modules
import os
import pickle
import sys
import time
import numpy as np
import pydub
import librosa
import glob
import gc

import tracemalloc



# ディレクトリのwavをすべてハッシュ化

def all2hash_melspec(dirpath):
    lists=[]
    for filename in os.listdir(dirpath):
        base,ext= os.path.splitext(filename)
        path=f"{dirpath}/{filename}"
        if ext=='.wav':
            spec,fs=modules.path2sgram(path)
            yx=modules.detect_peaks(spec, filter_size=30, order=0.3)
            hash,pairs=modules.peaks2hash(yx[1],yx[0])
            lists.append([hash,base])
    return lists

def all2hash_cqt(dirpath):
    
    lists=[]
    for filename in os.listdir(dirpath):
        base,ext= os.path.splitext(filename)
        path=f"{dirpath}/{filename}"
        if ext=='.wav':
            spec,fs=modules.path2cqt2sgram(path)
            x,y=modules.sgram2peaks(spec,amp_min=40,plot=False,PEAK_NEIGHBORHOOD_SIZE=8)
            hash,pairs=modules.peaks2hash(x,y)
            lists.append([hash,base])
    return lists

def put_all_sgram(dirpath):
    tracemalloc.start(15)
    n_put=3
    lists=[]
    dirpath=glob.glob(os.path.join(dirpath,"*.wav"))
    num_tuples=(88,)*n_put
    for i,j,k in np.ndindex(num_tuples):
        i_len_list=path2len_list(dirpath[i])
        j_len_list=path2len_list(dirpath[j])
        for i_t in i_len_list:
            mixing_data=modules.path2audiomixing(dirpath[i],dirpath[j],i_t)
            for j_t in j_len_list:
                mixing_data_=modules.wavpath2audiomixing(mixing_data,dirpath[j],j_t)
                mixing_data_=modules.audiosegment2librosawav(mixing_data_)
                
                sgram,wave,fs=modules.wav2sgram(mixing_data_,mixing_data.frame_rate)
                spec_db=librosa.power_to_db(sgram,ref=0)

                # 振幅に変換
                spec = np.abs(spec_db)
                
                
                yx=modules.detect_peaks(spec, filter_size=10, order=0.7)
                hash,pairs=modules.peaks2hash(yx[1],yx[0])
                base=f"{dirpath[i]}_{dirpath[j]}_{dirpath[k]}"
                lists.append([hash,base])
                print(modules.peaks2hash.__sizeof__())
                del hash
                del base
                del pairs
                gc.collect()
                
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)
                    for line in stat.traceback.format():
                        print(line)
                    print("=====")
    return lists

def savelist(listname,name):
    with open(name,'wb') as p:
        pickle.dump(listname,p)

def path2len_list(path):
    i_wav=pydub.AudioSegment.from_file(path)    
    i_len=i_wav.duration_seconds
    i_len=i_len-1
    i_data=np.arange(0,i_len,100)
    i_data=np.round(i_data,1)
    return i_data

def checkargs(args):
    if args!=1:
        print("pls only path dir")
        sys.exit()
    elif type(args)!=str:
        print("pls only path dir")
        sys.exit()

if __name__=="__main__":
    
    t1=time.perf_counter()
    lists=all2hash_melspec("./data/JKspeech-v_1_0/JKspeech")
    savelist(lists,"db/lists_db.pkl")
    t2=time.perf_counter()
    print(f"{t2-t1},,,finished")
    

    """
    t1=time.perf_counter()
    lists=put_all_sgram("./data/JKspeech-v_1_0/JKspeech")
    savelist(lists,"db/lists_db.pkl")
    t2=time.perf_counter()
    print(f"{t2-t1},,,finished")
    """