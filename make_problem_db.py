import modules
import os
import pickle
import sys

# ディレクトリのwavをすべてハッシュ化

def all2hash(dirpath):
    hashlist=[]
    for filename in os.listdir(dirpath):
        hash=modules.path2hash(filename)
        hashlist.append(hash)
    return hashlist

def savehashlist(hashlist):
    with open('problem_db.pkl','wb') as p:
        pickle.dump(hashlist,p)

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
    savehashlist(all2hash(args))