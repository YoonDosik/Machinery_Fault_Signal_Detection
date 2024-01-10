import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import time

def Creation_of_PCA_Train(train_loader):

    train_data = train_loader.dataset.data
    train_label = train_loader.dataset.label

    train_data = train_data.cpu().detach().numpy()
    train_label = train_label.cpu().detach().numpy()

    train_data = train_data.reshape(train_data.shape[0], -1)

    # 주성분 분석
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scale = scaler.transform(train_data)

    # pca = PCA(n_components = 0.95) # 주성분을 몇개로 할지 결정
    pca = PCA(n_components = 16) # 주성분을 몇개로 할지 결정
    printcipalComponents = pca.fit_transform(train_data_scale)
    pca.explained_variance_ratio_.cumsum()
    train_data = np.array(pd.DataFrame(data=printcipalComponents))

    return train_data, train_label, scaler

def Creation_of_PCA_Test(train_data,test_loader, scaler):

    test_data = test_loader.dataset.data
    test_label = test_loader.dataset.label

    test_data = test_data.cpu().detach().numpy()
    test_label = test_label.cpu().detach().numpy()

    test_data = test_data.reshape(test_data.shape[0], -1)

    test_data_scale = scaler.transform(test_data)

    pca = PCA(n_components=train_data.shape[1])  # 주성분을 몇개로 할지 결정
    printcipalComponents = pca.fit_transform(test_data_scale)
    pca.explained_variance_ratio_.cumsum()
    test_data = np.array(pd.DataFrame(data=printcipalComponents))

    return test_data, test_label

def Kernel_Density_Estimation_Train(args, train_data):

    start_time = time.time()
    params = {'bandwidth': np.logspace(- 4.5, 5, num=20, base=2)}

    # cv 교차 검증
    kde = GridSearchCV(KernelDensity(kernel= args.kernel), params , n_jobs = -1, cv = 5, verbose = 0)
    kde = kde.fit(train_data)

    test_time = time.time() - start_time

    print("best bandwidth: {0}".format(kde.best_estimator_.bandwidth))
    print("Triaining_time: {0}".format(test_time))

    return kde


def Kernel_Density_Estimation_Test(kde, test_data, test_label):

    test_score = (kde.score_samples(test_data))
    Test_ROC_value = (roc_auc_score(test_label, test_score) * 100)

    print('Testing_KDE_ROC AUC score: {:.2f}'.format(roc_auc_score(test_label, test_score)*100))

    return test_score, Test_ROC_value


def Isolation_Forest(train_data,test_data, test_label):

    # max_samples = 2500, 나머진 Default값
    start_time = time.time()

    clf = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 0.1, n_jobs = -1)
    clf.fit(train_data)

    scores = clf.decision_function(test_data)

    Test_ROC_Value = (roc_auc_score(test_label, scores) * 100)

    test_time = time.time() - start_time

    print('Testing_Isolation_Forest_ROC AUC score: {:.2f}'.format(roc_auc_score(test_label, scores) * 100))
    print('Testing_time_Isolation_Forest: {:.2f}'.format(test_time))

    return Test_ROC_Value

def OC_SVM(train_data, test_data, test_label, gam, Nu):

    start_time = time.time()

    clf = OneClassSVM(kernel='rbf', gamma= gam, nu = Nu)
    clf = clf.fit(train_data)

    scores = clf.score_samples(test_data)

    #MCDM 으로 *100 삭제함 --> 다시 *100으로 나중에 수정해야함
    Test_ROC_Value = (roc_auc_score(test_label, scores) * 100)

    test_time = time.time() - start_time

    print('Testing_SVDD_ROC AUC score: {:.2f}'.format(roc_auc_score(test_label, scores)*100))
    print('Testing_time_SVDD: {:.2f}'.format(test_time))

    return Test_ROC_Value
