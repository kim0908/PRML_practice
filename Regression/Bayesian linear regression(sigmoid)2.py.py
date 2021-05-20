# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:43:33 2021

@author: CSDSP_KIM
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Sigmoid_BasisFunc(x,M):
    features=np.zeros([len(x),1])
    for j in range(M):
        mu=(2*j)/M
        phy=np.zeros([len(x),1])
        for k in range(len(x)):
            tmp=(x-mu)
            phy[k]=1/(1+math.exp(-tmp[k]))
        features=np.concatenate((features,phy),axis=1)
    features=features[:,1:M+1]
    return features

if __name__ == '__main__':
    data = pd.read_csv("data_3.csv")
    X = data['X'].values
    T = data['T'].values
    alpha = 5*10 **-5
    beta = 10
    Num=np.array([2,4,25,50,70,100])
    M=9
    Total_feature= Sigmoid_BasisFunc(X,M)
    fig,axes=plt.subplots(2,3)
    plt.suptitle('Sigmoid_Basis')
    ax=np.array([axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]])
    standard=np.zeros([100,1])
    y_mean=np.zeros([100,1])
    for n in range(len(Num)):
        N=Num[n]
        X_train = X[0:N]
        T_train = T[0:N]
        x_feature = Sigmoid_BasisFunc(X_train,M)
        cov = np.linalg.inv(alpha*np.eye(M)+beta*x_feature.T.dot(x_feature))
        mean = beta*np.dot(cov,x_feature.T).dot(T_train)
        posterior_w = np.reshape(np.random.multivariate_normal(mean,cov,1),(1,M))
        
        y = Total_feature.dot(posterior_w.T)     
        for a in range(len(X)):
            standard[a] = np.sqrt((10 **-1)+Total_feature[a,:].T.dot(cov).dot(Total_feature[a,:]))
            y_mean [a] = mean.T.dot(Total_feature[a,:])
        idx = np.argsort(X) 
        xs = np.array(X)[idx] 
        ys = np.array(y_mean)[idx]
        standard= np.array(standard)[idx]
        
        ax[n].plot(xs,ys,color='b')
        ax[n].scatter(X_train,T_train,color='r')
        
        ys=np.squeeze(ys)
        standard=np.squeeze(standard)

        ax[n].fill_between(xs, ys-standard , ys+standard ,alpha=0.5,label='confidence interval')
        ax[n].set_title('N={}'.format(N))
        
        
        
