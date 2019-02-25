import glob
import os

for filename in glob.iglob('./timit/**/**/**/*.wav'):
     os.system('./sph2pipe -f wav '+filename+' '+filename.replace('.wav', 'CONVERTED.wav'))
