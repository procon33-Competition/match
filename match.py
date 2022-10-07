import binascii
import modules
import os
import sys
import pickle
import matplotlib.pyplot as plt
import shutil
import glob


def load_pkl(path):
    with open(path, 'rb') as p:
        l = pickle.load(p)
    return l

def match_db(audio_path,num,graph=False,scatter=False):
    """"""
        
    spec,fs=modules.path2sgram(audio_path)
    x,y=modules.sgram2peaks(spec,amp_min=50,plot=False)
    audio_hash,pairs=modules.peaks2hash(x,y)
    
    huga=[]
    piyo=[]
    hoges=[]
    hoges_=[]
    saga=dict()
    tmp=dict()
    lists=load_pkl("db/lists_db.pkl")
       
    for problem,name_ in lists:
        hoge,hoge_=modules.hashmatching(audio_hash,problem,graph=graph)
        #tmp[name_]=f"{hoge},{hoge_},{_hoge}
        tmp[name_]=hoge
        saga[name_]=problem
        piyo.append(hoge)
        huga.append([hoge,problem])
        hoges.append(hoge)
        hoges_.append(hoge_)

    plt.hist(piyo,bins=44)
    plt.savefig("tmp/ans/ans_hist.png")
    tmp=sorted(tmp.items(),key=lambda x:x[1],reverse=True)
    shutil.rmtree(f"tmp/ans/")
    os.makedirs(f"tmp/ans",exist_ok=True)
    key_s=tmp[:num]
    for key,_ in key_s:
        modules.hashmatching_img(audio_hash,saga[key],key)

    #単体のときは/
    #複数のやつのときは\\

    # audio_path=audio_path.split("\\")
    audio_path=audio_path.split("/")

    audio_path=f"{audio_path[-2]}_{audio_path[-1]}"
    if scatter==True:
        # shutil.rmtree(f"tmp/scatter/")
        # os.makedirs(f"tmp/scatter",exist_ok=True)
        plt.scatter(hoges,hoges_)
        plt.savefig(f"./tmp/scatter/{audio_path}.png")
        plt.close()
        print(f"{audio_path}")
    
    # for i in range(num):
    #     print(tmp[i])
    # return ans
    print(f"{audio_path}==={tmp[:num]}")

if __name__=="__main__":
    match_db("data/sample_Q_202205/sample_Q_202205/sample_Q_J01/problem1.wav",20,graph=False)
    
    # files=glob.glob("data\sample_Q_202205\sample_Q_202205\*\*.wav")

    # for file in files:
    #     #print(file)
    #     match_db(file,5,False,True)
    

        