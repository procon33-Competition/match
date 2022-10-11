import modules
import os
import pickle
import sys
import time
import numpy as np
import pydub
import glob


# ディレクトリのwavをすべてハッシュ化

def all2hash_melspec(dirpath):
    lists=[]
    for filename in os.listdir(dirpath):
        base,ext= os.path.splitext(filename)
        path=f"{dirpath}/{filename}"
        if ext=='.wav':
            spec,fs=modules.path2sgram(path)
            x,y=modules.sgram2peaks(spec,amp_min=40,plot=False,PEAK_NEIGHBORHOOD_SIZE=8)
            hash,pairs=modules.peaks2hash(x,y)
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
    n_put=3
    lists=[]
    dirpath=glob.glob(os.path.join(dirpath,"*.wav"))
    num_tuples=(88,)*n_put
    for i,j,k in np.ndindex(num_tuples):
        i_len_list=path2len_list(dirpath[i])
        j_len_list=path2len_list(dirpath[j])
        for i_t in i_len_list:
            mixing_data=modules.audio_mixing(dirpath[i],dirpath[j],i_t)
            for j_t in j_len_list:
                mixing_data_=modules.audio_mixing(dirpath[i],dirpath[j],j_t)
                mixing_data_.export("./tmp/mixing_data.wav",format="wav")
                del mixing_data_
                spec,fs=modules.path2sgram("./tmp/mixing_data.wav")
                x,y=modules.sgram2peaks(spec,amp_min=40,plot=False,PEAK_NEIGHBORHOOD_SIZE=8)
                hash,pairs=modules.peaks2hash(x,y)
                base=f"{dirpath[i]}_{dirpath[j]}_{dirpath[k]}"
                lists.append([hash,base])
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
    """
    t1=time.perf_counter()
    lists=all2hash_melspec("./data/JKspeech-v_1_0/JKspeech")
    savelist(lists,"db/lists_db.pkl")
    t2=time.perf_counter()
    print(f"{t2-t1},,,finished")
    """
    t1=time.perf_counter()
    put_all_sgram("./data/JKspeech-v_1_0/JKspeech")
    savelist(lists,"db/lists_db.pkl")
    t2=time.perf_counter()
    print(f"{t2-t1},,,finished")
    