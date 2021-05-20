# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:12:49 2021

@author: CSDSP_KIM
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Polynomial_basis_functions(x,Max_degree):
    x=np.reshape(x,(len(x),1))
    poly_features = np.ones(shape=(len(x),1))
    for i in range(0, Max_degree):
        features = x ** (i+1)
        poly_features = np.concatenate((poly_features, features), axis=1)
    return poly_features

def rms_error(x,w,target):
    tmp = [(x[i,:].dot(w)-target[i]) **2 for i in range(len(target))]
    loss=(0.5)*sum(tmp)
    rms_error = math.pow(((2*loss) / len(target))  ,0.5)
    return rms_error

def rms_error_add(x,w,target,lamda):
    tmp = [(x[i,:].dot(w)-target[i]) **2 for i in range(len(target))]
    loss=(0.5)*sum(tmp)+(lamda/2)*w.dot(w)
    rms_error = math.pow(((2*loss) / len(target))  ,0.5)
    return rms_error

if __name__ == '__main__':
    data = pd.read_csv("data_1.csv")
    train_x = np.hsplit(data.iloc[:,0], [70])[0]
    test_x = np.hsplit(data.iloc[:,0], [70])[1]
    train_target = np.hsplit(data.iloc[:,1], [70])[0]
    test_target = np.hsplit(data.iloc[:,1], [70])[1]
    train_rms_error=np.zeros([10,1])
    train_rms_error_add_ru=np.zeros([1000,1])
    test_rms_error=np.zeros([10,1])
    test_rms_error_add_ru=np.zeros([1000,1])
    ln_lamda = np.linspace(-30, -5, num=1000)
    lamda = np.exp(ln_lamda)
    for m in range(0,10):
        Max_degree=m
        w=np.zeros([Max_degree,1])
        print('Max_degree = {}'.format(Max_degree))
        poly_train_x = Polynomial_basis_functions(train_x.values,Max_degree)
        poly_test_x = Polynomial_basis_functions(test_x.values,Max_degree)
        w=np.linalg.inv(poly_train_x.T .dot(poly_train_x)).dot(poly_train_x.T).dot(train_target)
        #print('weight={}'.format(w))
        train_rms_error[m]=rms_error(poly_train_x,w,train_target.values)
        test_rms_error[m]=rms_error(poly_test_x,w,test_target.values)
        
    for a in range(0,len(lamda)):
        I=np.shape(poly_train_x.T .dot(poly_train_x))[0]
        w_add_regu = np.linalg.inv(lamda[a]*np.eye(I)+np.dot(poly_train_x.T,poly_train_x)).dot(poly_train_x.T).dot(train_target)
        train_rms_error_add_ru[a]=rms_error_add(poly_train_x,w_add_regu,train_target.values,lamda[a])
        test_rms_error_add_ru[a]=rms_error_add(poly_test_x,w_add_regu,test_target.values,lamda[a])
     
    plt.figure()
    plt.title('RMS error with different M')
    plt.plot(train_rms_error,label = 'train')
    plt.plot(test_rms_error,label = 'test')
    plt.legend()
    plt.xlabel("M")
    plt.xlim([1,9])
    
    plt.figure()
    plt.title('RMS error with different regularization coefficient(M=9)')
    plt.plot(ln_lamda,train_rms_error_add_ru,label = 'train_add')
    plt.plot(ln_lamda,test_rms_error_add_ru,label = 'test_add')
    plt.legend()
    plt.xlabel("Ln_lamda")
    plt.ylabel("Erms")
    plt.show()
    
