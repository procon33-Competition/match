import modules
import os
import sys
import pickle


def load_pkl(path):
    with open(path, 'rb') as p:
        l = pickle.load(p)
    return l

def match_db(audio_path,db_path="problem_db.pkl"):
    audio_hash=modules.path2hash(audio_path)
    problem_db=load_pkl(db_path)
    ans=[]
    i=0
    for problem_hash in problem_db:
        tmp=modules.hashmatching(audio_hash,problem_hash)
        ans.append([i,tmp])
        i+=1
    return ans

if __name__=="__main__":
    args=sys.argv
    ans=match_db(args)
    ans=sorted(ans,key=lambda x:x[1],reverse=True)
    print(ans)