# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:52:44 2021

@author: CSDSP_KIM
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def Basis_functions(x):
    tmp1 = np.zeros([len(x),4])
    tmp2= np.zeros([len(x),1])
    tmp3= np.zeros([len(x),1])
    for i in range(4):
        tmp1[:,i] = x[:,i]
        for m in range(i,4):
            t2 = x[:,i]*x[:,m]
            tmp2=np.c_[tmp2,t2]
            for k in range(m,4):
                t3=x[:,i]*x[:,m]*x[:,k]
                tmp3=np.c_[tmp3,t3]
    tmp2= np.delete(tmp2,0,axis=1)
    tmp3= np.delete(tmp3,0,axis=1)
    poly_features = np.hstack((np.ones([len(x),1]),tmp1,tmp2,tmp3))
    return poly_features

def rms_error(x,w,target):
    tmp = [(x[i,:].dot(w)-target[i]) **2 for i in range(len(target))]
    loss=(0.5)*sum(tmp)
    rms_error = math.pow(((2*loss) / len(target))  ,0.5)
    return rms_error

if __name__ == '__main__':
    data_X= loadmat("Iris_X.mat")
    data_T= loadmat("Iris_T.mat")
    X = data_X["X"]
    T = data_T["T"]

    test_T =np.vstack((T[0:50][40:50],T[50:100][40:50],T[100:150][40:50]))
    test_X = np.vstack((X[40:50,:],X[90:100,:],X[140:150,:]))
    train_T = np.vstack((T[0:50][0:40],T[50:100][0:40],T[100:150][0:40]))
    train_X = np.vstack((X[0:40,:],X[50:90,:],X[100:140,:]))

    train_features = Basis_functions(train_X)
    test_features = Basis_functions(test_X)
    w=np.matmul(np.matmul(np.linalg.inv(np.matmul(train_features.T,train_features)),train_features.T),train_T)
    
    plt.figure()
    predic_test=[(test_features[i,:].dot(w)) for i in range(len(test_T))]
    plt.plot(predic_test,label='Testing_predict')
    plt.plot(test_T,label='Testing_traget')
    plt.legend()
    
    train_rms_error=rms_error(train_features,w,train_T)
    test_rms_error=rms_error(test_features,w,test_T)
    print('train_rms_error={}'.format(train_rms_error))
    print('test_rms_error={}'.format(test_rms_error))

    
    
    
