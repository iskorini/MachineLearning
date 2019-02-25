from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import glob


for filename in glob.iglob('./timit/**/**/**/*CONVERTED.wav'):
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate, winstep=0.01, winlen=0.025, appendEnergy=True, samplerate=16000)
    np.savetxt(filename.replace('CONVERTED.wav', '-feature.txt'), mfcc_feat)

#import speechpy as sphp
#import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt

#(rate, sig) = wav.read('./timit/test/dr1/faks0/sa1CONVERTED.wav')
#mfcc_feat = sphp.feature.mfcc(sig, sampling_frequency=16000, frame_length=0.025 , frame_stride=0.01)
#derivative_mfcc_feat = sphp.feature.extract_derivative_feature(mfcc_feat)
#plt.figure()
#plt.plot(mfcc_feat)
#plt.show()
#plt.figure()

