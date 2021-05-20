# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:22:40 2021

@author: CSDSP_KIM
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Gaussian_BasisFunc(x,M):
    features=np.zeros([len(x),1])
    for j in range(M):
        mu=(j-4)/4
        phy=np.zeros([len(x),1])
        for k in range(len(x)):
            tmp=-0.5*np.multiply(x-mu,x-mu)
            phy[k]=math.exp(tmp[k])
        features=np.concatenate((features,phy),axis=1)
    features=features[:,1:M+1]
    return features

if __name__ == '__main__':
    data = pd.read_csv("data_3.csv")
    X = data['X'].values
    T = data['T'].values
    M=9
    alpha = 5*10 **-5
    beta = 10
    Num=np.array([2,4,25,50,70,100])
    Total_feature= Gaussian_BasisFunc(X,M)
    fig,axes=plt.subplots(2,3)
    plt.suptitle('Gaussian_Basis')
    ax=np.array([axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]])
    
    for n in range(len(Num)):
        for h in range(5):
            N=Num[n]
            X_train = X[0:N]
            T_train = T[0:N]
            x_feature = Gaussian_BasisFunc(X_train,M)
            cov = np.linalg.inv(alpha*np.eye(M)+beta*x_feature.T.dot(x_feature))
            mean = beta*np.dot(cov,x_feature.T).dot(T_train)
            posterior_w = np.reshape(np.random.multivariate_normal(mean,cov,1),(1,M))
            
            y = Total_feature.dot(posterior_w.T)
            
            idx = np.argsort(X) 
            xs = np.array(X)[idx] 
            ys = np.array(y)[idx]
            XX = np.array(X)[idx]
            TT= np.array(T)[idx]
            ax[n].plot(xs,ys,color='b')
            
        ax[n].scatter(X_train,T_train,color='r')
        ax[n].set_title('N={}'.format(N))
            
            
