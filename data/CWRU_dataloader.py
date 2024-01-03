
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('C:\\Users\\com\\PycharmProjects\\Machinery Fault Signal Detection\\data')
import CWRU_preprocessing as st
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

    type = path[24:-11]

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

def Normal_Train_test_spliting(path,percent, sampling_rate, wd):

    'percent : 학습 및 테스트 데이터의 비율'

    start_time = time.time()

    Normal_data = st.Signal_data(path, sampling_rate, wd)

    Normal_true_label, Normal_sign_label = Creation_label(Normal_data, path)

    Normal_train, Normal_test, Normal_train_label, Normal_test_label = train_test_split(Normal_data, Normal_sign_label, test_size = 1 - percent, shuffle = True)

    test_time = time.time() - start_time

    print('Spend Time : {:.3f}'. format(test_time))

    return Normal_train, Normal_test, Normal_train_label, Normal_test_label

def Outlier_data_Creation(path, sampling_rate, wd):

    start_time = time.time()

    out_data = st.Signal_data(path, sampling_rate, wd)

    true_label, sign_label = Creation_label(out_data, path)

    test_time = time.time() - start_time

    print('Spend Time : {:.3f}'. format(test_time))

    return out_data, true_label, sign_label

def get_dataloader(args, Normal_path, sampling_rate, wd):

    train_data, test_data, train_label, test_label = Normal_Train_test_spliting(Normal_path, args.percent, sampling_rate, wd)

    data_train = data_loader(train_data, train_label)

    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle = True, num_workers= 0)

    return test_data, test_label, dataloader_train

def test_get_dataloader(path, sampling_rate, wd):

    out_test_data, out_test_true_label, out_test_sign_label = Outlier_data_Creation(path, sampling_rate, wd)

    return out_test_data, out_test_sign_label



