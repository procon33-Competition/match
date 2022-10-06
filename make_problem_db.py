from unicodedata import name
import modules
import os
import pickle
import sys
import time

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
            spec,fs=modules.path2sgram(path)
            x,y=modules.sgram2peaks(spec,amp_min=40,plot=False,PEAK_NEIGHBORHOOD_SIZE=8)
            hash,pairs=modules.peaks2hash(x,y)
            lists.append([hash,base])
    return lists

def savelist(listname,name):
    with open(name,'wb') as p:
        pickle.dump(listname,p)

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