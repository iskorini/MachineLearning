import glob

freq = dict()

for filename in glob.iglob('./timit/**/**/**/*.phn'):
    file = open(filename, 'r').read()
    for lines in file.splitlines():
        ph = lines.split(" ")[2]
        if ph not in freq.keys():
            freq[ph] = 0
        freq[ph] = freq[ph] + 1
sorted_freq = sorted(freq.items(), key=lambda kv: kv[1])
print(sorted_freq)
