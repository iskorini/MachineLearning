from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import glob

for filename in glob.iglob('./timit/**/**/**/*CONVERTED.wav'):
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate, winstep=0.01, winlen=0.025, appendEnergy=True, samplerate=16000)
    np.savetxt(filename.replace('CONVERTED.wav', '-feature.txt'), mfcc_feat)
