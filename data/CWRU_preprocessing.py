

import pandas as pd
import os
import numpy as np
import librosa
import random

def Load_data_from_folder(path):

    csv_list = [file_name for file_name in os.listdir(path) if file_name.endswith(".csv")]
    ID_list = [csv[-10:-4] for csv in csv_list]

    data_list = []

    for idx, csv in enumerate(csv_list):

        df = pd.read_csv(os.path.join(path, csv), encoding="cp949", header=0, usecols=[1])
        data_list.append(df)

    data_tuple = list(zip(ID_list, data_list))

    return data_tuple

def Convert_to_Matrix(data_list):

    for id, data in data_list:

        vib_list = [data.values[:, 0].reshape(1, -1) for id, data in data_list]

    shape_list = []

    for i, value in enumerate(vib_list):

        shape_list.append(value.shape[1])

    min_shape = np.min(np.array(shape_list))

    slicing_data = []

    for i in range(len(vib_list)):
        slicing_data.append(vib_list[i][:,:min_shape])

    shape_set = set([])
    for idx, signal in enumerate(slicing_data):
        shape_set.add(signal.shape)
    assert len(shape_set)==1, "data 중에 길이가 다른 것이 있습니다."

    vib_signal_matrix = np.concatenate(slicing_data, axis=0)

    return vib_signal_matrix

def Sampling_Unit_Signal(sig_mat, sr, wd):

    # sr : sample ratio --> 10240, wd : window_size --> 비율 0.25 (wd = 4)

    ex_num, ex_len = sig_mat.shape
    # rb = bound * sr # 단위 샘플링 범위 ex) 2초에서
    # rl = length * sr # 단위 샘플링 길이 ex) 1초 샘플링

    sampled = []
    #
    for i in range(ex_num):
        for j in range(int((ex_len/sr)*wd)):
            sampled.append(sig_mat[i][int((sr*(1/wd)*j)):int(((sr*(1/wd))*j)+sr)])

    a = np.zeros((len(sampled),sr,1))

    for i in range(len(sampled)):

        if np.array(sampled[i]).shape[0] == sr:

            a[i] = np.array(sampled[i]).reshape(-1,1)

    return a

def Signal_data(path, sampling_rate, wd):

    data_tuple = Load_data_from_folder(path)
    vib_signal_matrix = Convert_to_Matrix(data_tuple)
    data = Sampling_Unit_Signal(vib_signal_matrix, sampling_rate, wd)

    return data
