
import sys
import pandas as pd
import os
import random
import numpy as np
import importlib
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\data')
import paderborn_dataloader as PP
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\networks')
import deepsvdd_paderborn as Deep_SVDD

importlib.reload(Deep_SVDD)

class Args:

    # Deep SVDD 파라미터

    ''':param
    'num_epochs : epoch of fine-tuning'
    'num_epochs_ae : epoch of pre-training'
    'lr : learning rate of fine-tuning'
    'lr_milestones : heaviside of fine-tuning'
    'lr_ae : learning rate of pre-training'
    'lr_milestones_ae : learning rate of pre-training'
    'latent_dim : dimension of latent space :'
    '''

    num_epochs = 200
    num_epochs_ae = 6
    patience = 50
    lr = 0.001
    weight_decay = 0.5e-6
    weight_decay_ae = 0.5e-3
    lr_ae = 0.001
    lr_milestones = [100]
    lr_milestones_ae = [3]
    batch_size = 128
    pretrain = True
    latent_dim = 16
    save_path = 'D:\\Python_Code'
    percent = 0.8

# Experiment scenario for paderborn
class Scenario_0_K003:

    # Hyperparameter
    sampling_rate = 2048
    percent = 0.8
    num_scenario = 0
    # K003 & Scenario_0
    Normal_path = 'D:\\PADERBORN\\Normal\\K003'
    AIRD_path = 'D:\\PADERBORN\\AIRD\\Scenario_1'
    AORD_path = 'D:\\PADERBORN\\AORD\\Scenario_1'
    RIRD_path = 'D:\\PADERBORN\\RIRD\\Scenario_1'
    RORD_path = 'D:\\PADERBORN\\RORD\\Scenario_1'

class Scenario_1_K003:

    # Hyperparameter
    sampling_rate = 2048
    percent = 0.8
    num_scenario = 1
    # K003 & Scenario_0
    Normal_path = 'D:\\PADERBORN\\Normal\\K003'
    AIRD_path = 'D:\\PADERBORN\\AIRD\\Scenario_2'
    AORD_path = 'D:\\PADERBORN\\AORD\\Scenario_2'
    RIRD_path = 'D:\\PADERBORN\\RIRD\\Scenario_2'
    RORD_path = 'D:\\PADERBORN\\RORD\\Scenario_2'

class Scenario_2_K003:
    # Hyperparameter
    sampling_rate = 2048
    percent = 0.8
    num_scenario = 2
    # K003 & Scenario_0
    Normal_path = 'D:\\PADERBORN\\Normal\\K003'
    AIRD_path = 'D:\\PADERBORN\\AIRD\\Scenario_3'
    AORD_path = 'D:\\PADERBORN\\AORD\\Scenario_3'
    RIRD_path = 'D:\\PADERBORN\\RIRD\\Scenario_3'
    RORD_path = 'D:\\PADERBORN\\RORD\\Scenario_3'

class Scenario_3_K003:
    # Hyperparameter
    sampling_rate = 2048
    percent = 0.8
    num_scenario = 3
    # K003 & Scenario_0
    Normal_path = 'D:\\PADERBORN\\Normal\\K003'
    AIRD_path = 'D:\\PADERBORN\\AIRD\\Scenario_4'
    AORD_path = 'D:\\PADERBORN\\AORD\\Scenario_4'
    RIRD_path = 'D:\\PADERBORN\\RIRD\\Scenario_4'
    RORD_path = 'D:\\PADERBORN\\RORD\\Scenario_4'

args = Args()

scenario_0_K003_args = Scenario_0_K003()
scenario_1_K003_args = Scenario_1_K003()
scenario_2_K003_args = Scenario_2_K003()
scenario_3_K003_args = Scenario_3_K003()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def DataLoader_Function(args, scenario_0_args):

    ''':param
    args : Experiment parameter
    scenario_0_args : CWRU Case
    N_0_DE --> sampling_rate = 12000
    N_1_DE, N_2_DE, N_3_DE  --> sampling_rate = 48000
    '''

    Normal_test_data, Normal_test_label, dataloader_train = PP.get_dataloader(args,scenario_0_args, scenario_0_args.Normal_path, scenario_0_args.sampling_rate)

    Path_list = []

    Path_list.append(scenario_0_args.AIRD_path)
    Path_list.append(scenario_0_args.AORD_path)
    Path_list.append(scenario_0_args.RIRD_path)
    Path_list.append(scenario_0_args.RORD_path)

    # Anomaly Data 불러오기

    Anomaly_Test_data = []
    Anomaly_Test_label = []

    for i in Path_list:

        out_test_data, out_test_sign_label = PP.test_get_dataloader(i, scenario_0_args.sampling_rate)
        Anomaly_Test_data.append(out_test_data)
        Anomaly_Test_label.append(out_test_sign_label)

    data_test = np.concatenate(Anomaly_Test_data)
    data_test = np.vstack([Normal_test_data[scenario_0_args.num_scenario],data_test])

    data_label = np.concatenate(Anomaly_Test_label)
    data_label = np.vstack([Normal_test_label[scenario_0_args.num_scenario],data_label])

    data_test = PP.data_loader(data_test, data_label)

    dataloader_test = PP.DataLoader(data_test, batch_size=args.batch_size, shuffle = True, num_workers= 0)

    return dataloader_train, dataloader_test

def Comparison_Experiment_DE_48(args, device, scenario_0_args, epoch):

    AUC_Result = np.zeros((epoch,1))

    # epoch = 반복 실험 횟수
    for i in range(epoch):

        print(str(i)+ "번쨰 실험 진행중입니다 ============")

        dataloader_train, dataloader_test = DataLoader_Function(args, scenario_0_args)

        # Deep One Class SVDD

        Deep_SVDD.pretrain(args, dataloader_train, device)
        deep_one_net, deep_one_c, z, Deep_SVDD_test_time = Deep_SVDD.train(args, dataloader_train, device)
        Deep_SVDD_labels, Deep_SVDD_scores, Deep_SVDD_ROC_value, Deep_SVDD_z = Deep_SVDD.evaluation(deep_one_net, deep_one_c, dataloader_test,device)
        AUC_Result[i][0] = Deep_SVDD_ROC_value

    a = pd.DataFrame(AUC_Result)
    a.columns = ['Deep_svdd']
    # save path of results
    a.to_csv(str(scenario_0_args.Normal_path[-4:] + "_" + str(scenario_0_args.num_scenario) + "_AUC_Result.csv"))

# Epoch of experiments
epoch = 1

# K003
AUC_Result_DE_K003_0 = Comparison_Experiment_DE_48(args, device, scenario_0_K003_args, epoch)
AUC_Result_DE_K003_1 = Comparison_Experiment_DE_48(args, device, scenario_1_K003_args, epoch)
AUC_Result_DE_K003_2 = Comparison_Experiment_DE_48(args, device, scenario_2_K003_args, epoch)
AUC_Result_DE_K003_3 = Comparison_Experiment_DE_48(args, device, scenario_3_K003_args, epoch)
