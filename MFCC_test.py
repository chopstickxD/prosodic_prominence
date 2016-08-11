from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import preprocessing as read


#dataSet = read.readAudio('/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/CNE/*16k.wav')
#data = read.getElement(dataSet, '000001-prompt', 2)
#print(dataSet)
rate, sig = wav.read('/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/CNE/000056-correction_16k.wav')
#sig = data[1]
#rate = data[0]
'''
max_length = 48000
if(len(sig)>max_length):
    sig = sig[0:max_length]
else:
    sig = np.pad(sig, (0, max_length - len(sig)), 'constant', constant_values=(0, 0))
'''

#print(sig)
#sig = sig[0:4000]
#print(len(sig))
mfcc_feat = mfcc(sig, rate)
#print(len(mfcc_feat))
#print(len(mfcc_feat))
#fbank_feat = logfbank(sig, rate)



#print(mfcc_feat.shape[0])
#print(rate)
#print('abhier mfcc all \n', mfcc_feat, '\n')
#print('erster mfcc \n', mfcc_feat[0])
#print('zeilen: ', mfcc_feat.shape[0], 'spalten: ', mfcc_feat.shape[1])

plt.subplot(312)
plt.magnitude_spectrum(sig)

plt.subplot(311)

plt.plot(sig)
plt.xlabel('time')
plt.ylabel('amplitude')


plt.subplot(313)

numCoef = np.arange(1, mfcc_feat.shape[1]+1)
markerline, stemlines, baseline = plt.stem(mfcc_feat[0], '-')
plt.setp(markerline, linewidth=2, color='b')
plt.setp(stemlines, linewidth=2, color='b')
plt.setp(baseline, linewidth=2, color='b')



plt.show()
plt.close()
