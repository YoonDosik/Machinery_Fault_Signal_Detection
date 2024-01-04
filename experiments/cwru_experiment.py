
import sys
import pandas as pd
import os
import numpy as np
import importlib
import torch

sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\data')
import CWRU_dataloader as cwp
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
import deepsvdd_cwru as Deep_SVDD
import autoencoder_cwru as AE

# Anomaly detection scenario for CWRU dataset

class Scenario_0_FE:

    # 0.01초 --> 120 (sampling rate 12000인 경우)
    sampling_rate_12 = 1024
    # 0.01초 --> 480 (sampling rate 48000인 경우)
    sampling_rate_48 = 1024
    # window ratio
    wd = 4

    Normal_path = 'D:\\CASE_WESTERN_RESERVE\\Normal\\Pre_N_0_FE'
    out_0_path = 'D:\\CASE_WESTERN_RESERVE\\Outer\\Out_FE\\0_FE'
    inn_0_path = 'D:\\CASE_WESTERN_RESERVE\\Inner\\Inner_FE\\0_FE'
    bal_0_path = 'D:\\CASE_WESTERN_RESERVE\\Ball\\Ball_FE\\0_FE'

class Scenario_1_FE:

    # 0.01초 --> 120 (sampling rate 12000인 경우)
    sampling_rate_12 = 1024
    # 0.01초 --> 480 (sampling rate 48000인 경우)
    sampling_rate_48 = 1024
    # window ratio
    wd = 4

    Normal_path = 'D:\\CASE_WESTERN_RESERVE\\Normal\\Pre_N_1_FE'
    out_0_path = 'D:\\CASE_WESTERN_RESERVE\\Outer\\Out_FE\\1_FE'
    inn_0_path = 'D:\\CASE_WESTERN_RESERVE\\Inner\\Inner_FE\\1_FE'
    bal_0_path = 'D:\\CASE_WESTERN_RESERVE\\Ball\\Ball_FE\\1_FE'

class Scenario_2_FE:

    # 0.01초 --> 120 (sampling rate 12000인 경우)
    sampling_rate_12 = 1024
    # 0.01초 --> 480 (sampling rate 48000인 경우)
    sampling_rate_48 = 1024
    # window ratio
    wd = 4

    Normal_path = 'D:\\CASE_WESTERN_RESERVE\\Normal\\Pre_N_2_FE'
    out_0_path = 'D:\\CASE_WESTERN_RESERVE\\Outer\\Out_FE\\2_FE'
    inn_0_path = 'D:\\CASE_WESTERN_RESERVE\\Inner\\Inner_FE\\2_FE'
    bal_0_path = 'D:\\CASE_WESTERN_RESERVE\\Ball\\Ball_FE\\2_FE'

class Scenario_3_FE:

    # 0.01초 --> 120 (sampling rate 12000인 경우)
    sampling_rate_12 = 1024
    # 0.01초 --> 480 (sampling rate 48000인 경우)
    sampling_rate_48 = 1024
    # window ratio
    wd = 4

    Normal_path = 'D:\\CASE_WESTERN_RESERVE\\Normal\\Pre_N_3_FE'
    out_0_path = 'D:\\CASE_WESTERN_RESERVE\\Outer\\Out_FE\\3_FE'
    inn_0_path = 'D:\\CASE_WESTERN_RESERVE\\Inner\\Inner_FE\\3_FE'
    bal_0_path = 'D:\\CASE_WESTERN_RESERVE\\Ball\\Ball_FE\\3_FE'

# FE에 부착된 센서로부터 수집된 신호 데이터

scenario_0_fe_args = Scenario_0_FE()
scenario_1_fe_args = Scenario_1_FE()
scenario_2_fe_args = Scenario_2_FE()
scenario_3_fe_args = Scenario_3_FE()

def DataLoader_Function_FE(args,scenario_0_fe_args):

    ''':param
    args : Experiment parameter
    scenario_0_args : CWRU Case
    '''

    Normal_test_data, Normal_test_label, dataloader_train = cwp.get_dataloader(args, scenario_0_fe_args.Normal_path, scenario_0_fe_args.sampling_rate_12, scenario_0_fe_args.wd)

    Path_list_12 = []

    Path_list_12.append(scenario_0_fe_args.out_0_path)
    Path_list_12.append(scenario_0_fe_args.inn_0_path)
    Path_list_12.append(scenario_0_fe_args.bal_0_path)

    # Anomaly Data 불러오기
    Anomal_Test_data_12 = []
    Anomal_Test_label_12 = []

    for i in Path_list_12:

        out_test_data, out_test_sign_label = cwp.test_get_dataloader(i, scenario_0_fe_args.sampling_rate_12, scenario_0_fe_args.wd)

        Anomal_Test_data_12.append(out_test_data)
        Anomal_Test_label_12.append(out_test_sign_label)

    data_test = np.concatenate(Anomal_Test_data_12)
    data_test = np.vstack([Normal_test_data,data_test])

    data_label = np.concatenate(Anomal_Test_label_12)
    data_label = np.vstack([Normal_test_label,data_label])

    data_test = cwp.data_loader(data_test, data_label)

    dataloader_test = cwp.DataLoader(data_test, batch_size=args.batch_size, shuffle = True, num_workers= 0)

    return dataloader_train, dataloader_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:

    num_epochs = 200
    num_epochs_ae = 6
    patience = 50
    lr = 0.001
    weight_decay = 0.5e-6
    weight_decay_ae = 0.5e-3
    lr_ae = 0.001
    lr_milestones = [100]
    lr_milestones_ae = [3]
    batch_size = 32
    pretrain = True
    latent_dim = 16
    save_path = 'D:\\Python_Code'
    percent = 0.8
    k = 5

args = Args()

def Comparison_Experiment_FE(args, device, scenario_0_args, epoch):

    AUC_Result = np.zeros((epoch,2))

    # epoch = 반복 실험 횟수
    for i in range(epoch):

        dataloader_train, dataloader_test = DataLoader_Function_FE(args, scenario_0_args)

        # Deep One Class SVDD

        Deep_SVDD.pretrain(args, dataloader_train, device)
        deep_one_net, deep_one_c, z, Deep_SVDD_test_time = Deep_SVDD.train(args, dataloader_train, device)
        Deep_SVDD_labels, Deep_SVDD_scores, Deep_SVDD_ROC_value, Deep_SVDD_z = Deep_SVDD.evaluation(deep_one_net, deep_one_c, dataloader_test, device)
        AUC_Result[i][1] = Deep_SVDD_ROC_value


    a = pd.DataFrame(AUC_Result)
    a.columns = ['AE','Deep_SVDD']
    a.to_csv(str(scenario_0_args.Normal_path[-6:] + "_" + str(args.lr) + "_AUC_Result.csv"))

# 실험 반복 횟수
epoch = 1

AUC_Result_FE_0 = Comparison_Experiment_FE(args, device, scenario_0_fe_args, epoch)
AUC_Result_FE_1 = Comparison_Experiment_FE(args, device, scenario_1_fe_args, epoch)
AUC_Result_FE_2 = Comparison_Experiment_FE(args, device, scenario_2_fe_args, epoch)
AUC_Result_FE_3 = Comparison_Experiment_FE(args, device, scenario_3_fe_args, epoch)