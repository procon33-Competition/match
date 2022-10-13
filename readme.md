# アルゴリズム部

# 操作(9/30現在)

requiments.txt入れる
公式サイトからデータを持ってきて回答して/dataに配置する  
make_problem_db.pyを動かす  
match.pyのmatch_dbの引数にwavファイルのパスと読まれてる数を入れる

# ファイル説明
fig.py : 画像生成スクリプト  
make_problem_db.py : ハッシュdb作成スクリプト  
match.py : 分割データに対してどの問題データが使用されているかを推定するスクリプト  
run.py : 本番利用用のスクリプト 未完  
modules.py : モジュールをまとめたファイル  
test.py : 実験用のファイル  
test.ipynb : 実験用のnotebook  

# ディレクトリ構造

```
data  
  ├JKspeech-v_1_0  
    └JKspeech  
        └E01.wav  
        └etc.........  
  └sample_Q_202205  
        └sample_Q_202205  
            └sample_Q_E01  
                └problem.wav  
                └problem1.wav  
                └problem2.wav  
                └information.txt  
            └etc......  
db  
  └list_db.pkl  
```

### やってることメモ

* 音を時間を持つ画像にする(fftやらなんやらで)
* この画像のピーク、もしくは特徴点を見つける
* その点をまわりの点とペアリングする
* このペアリングした情報をハッシュテテーブに保存

* マッチングするときは上の手法して全探索 O(n)
* 実際の計算量はハッシュの数*問題音声の数(未確認)


### 将来実装

* ローカルのためにdockerfile
* ~~問題音声をすべてハッシュ化したやつのいい感の保存~~ 済
* 他のでピーク取得
* 高速化<-いる？
