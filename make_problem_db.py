from unicodedata import name
import modules
import os
import pickle
import sys

# ディレクトリのwavをすべてハッシュ化

def all2hash(dirpath):
    hashlist=[]
    namelist=[]
    lists=[]
    for filename in os.listdir(dirpath):
        base,ext= os.path.splitext(filename)
        path=f"{dirpath}/{filename}"
        if ext=='.wav':
            spec,fs=modules.path2sgram(path)
            x,y=modules.sgram2peaks(spec,amp_min=40,plot=False,PEAK_NEIGHBORHOOD_SIZE=8)
            hash,pairs=modules.peaks2hash(x,y)
            hashlist.append(hash)
            namelist.append(base)
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
    lists=all2hash("./data/JKspeech-v_1_0/JKspeech")
    savelist(lists,"db/lists_db.pkl")
    print("finished")