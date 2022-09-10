from unicodedata import name
import modules
import os
import pickle
import sys

# ディレクトリのwavをすべてハッシュ化

def all2hash(dirpath):
    hashlist=[]
    for filename in os.listdir(dirpath):
        base,ext= os.path.splitext(filename)
        namelist=[]
        if ext=='.wav':
            hash=modules.path2hash(dirpath+filename)
            hashlist.append(hash)
            namelist.append(base)
    return hashlist,namelist

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
    args=sys.srgv
    checkargs(args)
    hashlist,namelist=all2hash(args)
    savelist(hashlist)
    savelist(namelist)