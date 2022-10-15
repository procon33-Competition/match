import modules
import warnings
warnings.simplefilter('ignore', Warning)
import pydub

a1=pydub.AudioSegment.from_wav("data/to_ans/problem2.wav")
a2=pydub.AudioSegment.from_wav("data/to_ans/problem3.wav")
a3=pydub.AudioSegment.from_wav("data/to_ans/problem4.wav")

a1=pydub.AudioSegment.from_wav("data/JKspeech/E01.wav")
a2=pydub.AudioSegment.from_wav("data/JKspeech/J02.wav")
a3=pydub.AudioSegment.from_wav("data/JKspeech/E13.wav")
a4=pydub.AudioSegment.from_wav("data/JKspeech/E24.wav")
a5=pydub.AudioSegment.from_wav("data/JKspeech/E32.wav")
a6=pydub.AudioSegment.from_wav("data/JKspeech/J06.wav")


ax=a1.overlay(a2,100)
ax=ax.overlay(a3,1000)
ax=ax.overlay(a4,1500)
ax=ax.overlay(a5,1500)
ax=ax.overlay(a6,1800)

ax=ax[200:1200]

ax.export("data/to_ans/problems.wav",format="wav")


modules.antiphase("data/to_ans/problem2.wav","data/JKspeech/")
# modules.antiphase("data/sample_Q_202205/sample_Q_E02/problem2.wav","data/JKspeech/")
