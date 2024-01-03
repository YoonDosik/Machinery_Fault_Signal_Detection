
import pandas as pd
import os
import numpy as np
import librosa
import random

def Load_data_from_folder(path):

    csv_list = [file_name for file_name in os.listdir(path) if file_name.endswith(".csv")]
    ID_list = [csv[:-4] for csv in csv_list]

    data_list = []

    for idx, csv in enumerate(csv_list):

        df = pd.read_csv(os.path.join(path, csv), encoding="cp949", header=0, index_col=0)
        data_list.append(df)

    data_tuple = list(zip(ID_list, data_list))

    return data_tuple

def Convert_to_Matrix(data_list):

    for id, data in data_list:
        print(data)

    vib_list = [data.values[:, 0].reshape(-1,1) for id, data in data_list]

    shape_set = set([])
    for idx, signal in enumerate(vib_list):
        shape_set.add(signal.shape)
    assert len(shape_set)==1, "data 중에 길이가 다른 것이 있습니다."

    return vib_list

def Sampling_Unit_Signal(sig_mat, sr):

    # sr : sample ratio --> 10240
    # rb = 2
    # rl = 1

    Ex_num = []
    Ex_len = []

    for i in range(len(sig_mat)):
        ex_num, ex_len = sig_mat[i].shape
        Ex_num.append(ex_num)
        Ex_len.append(ex_len)

    # rb = bound * sr # 단위 샘플링 범위 ex) 2초에서
    # rl = length * sr # 단위 샘플링 길이 ex) 1초 샘플링

    Sampled = []
    for i in range(len(sig_mat)):
        sampled = []
        for j in range(int(Ex_num[i]/sr)):
            sampled.append(sig_mat[i][sr*j:sr*(j+1)])
        Sampled.append(sampled)

    Pre_Sampled = []

    for i in range(len(Sampled)):
        Pre_Sampled.append(np.array(Sampled[i]).reshape(-1,sr,1))

    return Pre_Sampled

def Signal_data(path, sampling_rate):

    data_tuple = Load_data_from_folder(path)
    signal_matrix = Convert_to_Matrix(data_tuple)
    data = Sampling_Unit_Signal(signal_matrix, sampling_rate)

    return data
