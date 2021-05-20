# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:02:49 2021

@author: CSDSP_KIM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(a):
    y = []
    for n in range(len(a)):
        total = 0
        a_max=np.max(a[n])
        for k in range(3):
            total += np.exp(a[n,k]-a_max)
        for k in range(3):
            tmp = (np.exp(a[n,k]-a_max))/total
            if tmp == 0:
                tmp = tmp + 10**(-6)
            y.append(tmp)
    y = np.reshape(y,(-1,3))
    return y
   
def Hessian( x , y, M ):
    H11,H12,H13,H21,H22,H23,H31,H32,H33=np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M))
    for n in range(len(y)):
        phi2 = np.reshape(x[n,:],(-1,1))*np.reshape(x[n,:],(1,-1))
        H11 += y[n,0]*(1-y[n,0])* phi2
        H12 += y[n,1]*(0-y[n,0])* phi2
        H13 += y[n,2]*(0-y[n,0])* phi2
        H21 += y[n,0]*(0-y[n,1])* phi2
        H22 += y[n,1]*(1-y[n,1])* phi2
        H23 += y[n,2]*(0-y[n,1])* phi2
        H31 += y[n,0]*(0-y[n,2])* phi2
        H32 += y[n,1]*(0-y[n,2])* phi2
        H33 += y[n,2]*(1-y[n,2])* phi2
                       
    return H11,H12,H13,H21,H22,H23,H31,H32,H33

def gradient(x, y ,t,M):
    g1,g2,g3=np.zeros((1,M)),np.zeros((1,M)),np.zeros((1,M))
    for n in range(len(y)):
        g1 +=(y[n,0]-t[n,0])*x[n]
        g2 +=(y[n,1]-t[n,1])*x[n]
        g3 +=(y[n,2]-t[n,2])*x[n]
    return g1.reshape(M,),g2.reshape(M,),g3.reshape(M,)

def update(w,H,G,lr,M):
    W_new=np.zeros((M,1))
    for k in range(3):
        tmp=np.reshape(w[:,k],(-1,1))-lr*np.dot(np.linalg.pinv(H[k][:,0:M]),np.reshape(G[:,0],(-1,1)))
        -lr*np.dot(np.linalg.pinv(H[k][:,M:2*M]),np.reshape(G[:,1],(-1,1)))
        -lr*np.dot(np.linalg.pinv(H[k][:,2*M:3*M]),np.reshape(G[:,2],(-1,1)))
        W_new=np.hstack((W_new,tmp))
    W_new=W_new[:,1:4]
    return W_new   

def predict(w,x):
    predict=np.zeros((len(x),3))
    a = np.dot(w.T,x.T).T
    y = softmax(a)
    for n in range(len(y)):
        predict[n,np.argmax(y[n])] = 1.0
    return predict 


def Accuracy(y, target):
    corret=0
    for n in range(len(y)):
        pre = np.argmax(y[n,:])
        tar = np.argmax(target[n,:])
        if pre == tar:
            corret=corret+1
    return (corret/len(y))

def Cross_Eropy ( y ,t):
    cross_ero=0
    for n in range(len(y)):
        for k in range(3):
            cross_ero=cross_ero+t[n,k]*np.log(y[n,k])
    return -cross_ero

if __name__ == '__main__':
    lr=0.0008
    M=14
    w=np.zeros((M,3))
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    wine_data = pd.read_csv("wine_train.csv",header=None)
    wine_data = np.array(wine_data.sample(frac=1.0))

    K_fold=8
    RAW_train_data= wine_data[:,3:16]
    RAW_train_data = np.insert(RAW_train_data, 0, 1, axis=1)
    RAW_train_targets=wine_data[:,0:3]

    
    num_samples = len(RAW_train_data) // K_fold

    val_data = RAW_train_data[num_samples : -1]
    val_targets = RAW_train_targets[num_samples : -1]
    train_data = np.concatenate( 
                         [RAW_train_data[: 0],
                         RAW_train_data[num_samples :]],
                         axis = 0) 
    train_targets = np.concatenate(
                         [RAW_train_targets[: 0],
                         RAW_train_targets[num_samples :]],
                         axis = 0)
    
    CE_train = 1
    pre_CE_train = 0
    while(np.abs(CE_train-pre_CE_train)>0.005):
        pre_CE_train = CE_train
        CE_train=0
        CE_val =0
        
        a_train=np.dot(w.T,train_data.T).T # num*3
        y_train=softmax(a_train) # num*3
        a_val=np.dot(w.T,val_data.T).T
        y_valid=softmax(a_val)
        
        CE_train=Cross_Eropy(y_train ,train_targets)/len(y_train)
        train_loss.append(CE_train)
        CE_val=Cross_Eropy(y_valid ,val_targets)/len(y_valid)
        valid_loss.append(CE_val)
        
        H11,H12,H13,H21,H22,H23,H31,H32,H33= Hessian(train_data, y_train,M ) 
        g1,g2,g3= gradient(train_data, y_train , train_targets ,M)
        
        w[:,0]-= lr*(np.dot(np.linalg.pinv(H11),g1)+np.dot(np.linalg.pinv(H21),g2)+np.dot(np.linalg.pinv(H31),g3))
        w[:,1]-= lr*(np.dot(np.linalg.pinv(H12),g1)+np.dot(np.linalg.pinv(H22),g2)+np.dot(np.linalg.pinv(H32),g3))
        w[:,2]-= lr*(np.dot(np.linalg.pinv(H13),g1)+np.dot(np.linalg.pinv(H23),g2)+np.dot(np.linalg.pinv(H33),g3))
        
        train_acc.append(Accuracy(y_train, train_targets))
        valid_acc.append(Accuracy(y_valid, val_targets))
     
    plt.figure()
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Loss')
    plt.legend(['train', 'validation'])
    plt.show()
    
    plt.figure()
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'validation'])
    plt.show()
    
    TEST_data = np.array(pd.read_csv("wine_test.csv",header=None))
    TEST_data = np.insert(TEST_data, 0, 1, axis=1)

    Test_pre=predict(w,TEST_data)
    print('Test_prediction=\n{}'.format(Test_pre))