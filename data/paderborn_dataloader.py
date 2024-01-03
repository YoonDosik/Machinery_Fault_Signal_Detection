
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
# py파일의 경로 입력
import sys
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\data')
import paderborn_preprocessing as st
from sklearn.model_selection import train_test_split
import time

class data_loader(data.Dataset):

    """This class is needed to processing batches for the dataloader."""

    def __init__(self, data, label):

        self.data = torch.FloatTensor(data)
        # Permute 함수를 통해 데이터의 차원 순서를 변경해줌
        self.data = self.data.permute(0, 2, 1)
        self.label = torch.LongTensor(label)

    def __getitem__(self, index):

        """return transformed items."""

        x = self.data[index]
        y = self.label[index]

        return x, y

    def __len__(self):

        """number of samples."""

        return len(self.data)

def Creation_label(data, path):

    type = path[13:-5]

    True_label = np.zeros((len(data),1))
    Sign_label = np.zeros((len(data),1))

    if type == 'Normal':

        for i in range(len(data)):

            True_label[i] = 1
            Sign_label[i] = 1

    else:
        for i in range(len(data)):

            True_label[i] = 2
            Sign_label[i] = -1

    return True_label, Sign_label

def Normal_Train_test_spliting(path, percent, sampling_rate):

    # 'percent : 학습 및 테스트 데이터의 비율
    # sampling_rate : Raw signal 길이

    start_time = time.time()

    Normal_data = st.Signal_data(path, sampling_rate)

    Normal_True_Label = []
    Normal_Sign_Label = []
    for i in range(len(Normal_data)):
        Normal_true_label, Normal_sign_label = Creation_label(Normal_data[i], path)
        Normal_True_Label.append(Normal_true_label)
        Normal_Sign_Label.append(Normal_sign_label)

    Normal_Train = []
    Normal_Test = []
    Normal_Train_Label = []
    Normal_Test_Label = []

    for i in range(len(Normal_data)):
        Normal_train, Normal_test, Normal_train_label, Normal_test_label = train_test_split(Normal_data[i], Normal_Sign_Label[i], test_size = 1 - percent, shuffle = True)
        Normal_Train.append(Normal_train)
        Normal_Test.append(Normal_test)
        Normal_Train_Label.append(Normal_train_label)
        Normal_Test_Label.append(Normal_test_label)

    test_time = time.time() - start_time

    print('Spend Time : {:.3f}'. format(test_time))

    return Normal_Train, Normal_Test, Normal_Train_Label, Normal_Test_Label

def Outlier_Train_test_spliting(path, percent, sampling_rate):

    start_time = time.time()

    Normal_data = st.Signal_data(path, sampling_rate)

    Normal_true_label, Normal_sign_label = Creation_label(Normal_data, path)

    Normal_train, Normal_test, Normal_train_label, Normal_test_label = train_test_split(Normal_data, Normal_sign_label,
                                                                                        test_size=1 - percent,
                                                                                        shuffle=True)

    test_time = time.time() - start_time

    print('Spend Time : {:.3f}'.format(test_time))

    return Normal_train, Normal_test, Normal_train_label, Normal_test_label

def Outlier_data_Creation(path, sampling_rate):

    start_time = time.time()

    out_data = st.Signal_data(path, sampling_rate)

    out_data = np.array(out_data).reshape(-1,sampling_rate,1)

    true_label, sign_label = Creation_label(out_data, path)

    test_time = time.time() - start_time

    print('Spend Time : {:.3f}'. format(test_time))

    return out_data, true_label, sign_label

def get_dataloader(args, scenario_0_args, Normal_path, sampling_rate):

    train_data, test_data, train_label, test_label = Normal_Train_test_spliting(Normal_path, args.percent, sampling_rate)

    Data_train = []

    for i in range(len(train_data)):
        data_train = data_loader(train_data[i], train_label[i])
        Data_train.append(data_train)

    dataloader_train = DataLoader(Data_train[scenario_0_args.num_scenario], batch_size=args.batch_size, shuffle = True, num_workers= 0)

    return test_data, test_label, dataloader_train

def test_get_dataloader(path, sampling_rate):

    out_test_data, out_test_true_label, out_test_sign_label = Outlier_data_Creation(path, sampling_rate)

    return out_test_data, out_test_sign_label



