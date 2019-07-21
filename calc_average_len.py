import glob
import numpy as np 
import pandas as pd 
import progressbar
from statistics import mean
from multiprocessing.pool import Pool as ThreadPool


def generate_data_frame(path):
    data = pd.DataFrame()
    tot = len(glob.glob(path))
    bar = progressbar.ProgressBar(maxval=tot, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    i = 0
    bar.start()
    data = []
    for file_name in glob.iglob(path):
        data.append(pd.read_csv(file_name))
        i = i+1
        bar.update(i)
    bar.finish()
    return data

if __name__ == "__main__":
    args = ['./preprocessed_dataset_LIBROSA/train2/**/**/**.csv','./preprocessed_dataset_LIBROSA/validation/**/**/**.csv']
    pool = ThreadPool(2)
    results = pool.map(generate_data_frame, args)
    pool.close()
    pool.join()        
    data_train = results[0]
    data_validation = results[1]
    data_train_mean = mean(map(lambda x: len(x), data_train))
    data_validation_mean = mean(map(lambda x: len(x), data_validation))
    avg = ((len(data_train)*data_train_mean) + (len(data_validation)*data_validation_mean))/(len(data_train)+len(data_validation))
    print("Average sentences length:")
    print(avg)

    