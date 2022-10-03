import binascii
import modules
import os
import sys
import pickle
import matplotlib.pyplot as plt


def load_pkl(path):
    with open(path, 'rb') as p:
        l = pickle.load(p)
    return l

def match_db(audio_path,num,english=True):
    '''
    Matching the audio to the database
    
    Args:
        audio_path (str): path to the audio file
        num (int): number of audio files to match
        name_path (str): path to the name file
        db_path (str): path to the database file
    
    '''

    
    spec,fs=modules.path2sgram(audio_path)
    x,y=modules.sgram2peaks(spec,amp_min=50,plot=False)
    audio_hash,pairs=modules.peaks2hash(x,y)
    
    huga=[]
    piyo=[]
    saga=dict()
    tmp=dict()
    if english==True:
        lists=load_pkl("db/E/lists_db.pkl")

        for problem,name in lists:
            hoge,hoge_=modules.hashmatching(audio_hash,problem)
            tmp[name]=f"{hoge},{hoge_}"
            saga=[name]=problem
            piyo.append(hoge)
            huga.append([hoge,problem])


    if english==False:
        lists=load_pkl("db/J/lists_db.pkl")
       
        for problem,name in lists:
            hoge,hoge_=modules.hashmatching(audio_hash,problem)
            tmp[name]=f"{hoge},{hoge_}"
            saga[name]=problem
            piyo.append(hoge)
            huga.append([hoge,problem])

    plt.hist(piyo,bins=44)
    plt.savefig("tmp/ans/ans_hist.png")
    tmp=sorted(tmp.items(),key=lambda x:x[1],reverse=True)

    key_s=tmp[:num]
    for key,_ in key_s:
        modules.hashmatching_img(audio_hash,saga[key],key)
    
    
    # for i in range(num):
    #     print(tmp[i])
    # return ans
    print(tmp[:num])

if __name__=="__main__":
    # match_db("data/sample_Q_202205/sample_Q_202205/sample_Q_E01/problem1.wav",3)
    match_db("data/sample_Q_202205/sample_Q_202205/sample_Q_J01/problem1.wav",5,False)