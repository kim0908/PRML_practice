# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:17:37 2021

@author: CSDSP_KIM
"""
import cvxopt
import cvxopt.solvers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaussian_kernel(x_1,x_2,sigma):
    return np.exp(-(np.linalg.norm(np.expand_dims(x_1, axis=1) - x_2, axis=-1) ** 2)/(2*sigma))

def SVM(x,y,sigma,C):
    x_len = len(x) # Amount of data
    dim = len(x[0])
    y = np.reshape(y, (-1, 1)) #30x1
    kernel = gaussian_kernel(x, x, sigma)
    P = cvxopt.matrix(np.dot(y, y.T) * kernel) #30x30
    q = cvxopt.matrix(-np.ones(x_len))
    #G = cvxopt.matrix(-np.eye(x_len))
    G = cvxopt.matrix(np.concatenate([-np.eye(x_len), np.eye(x_len)]))
    #h = cvxopt.matrix(np.zeros(x_len))
    h = cvxopt.matrix(np.concatenate([np.zeros(x_len), np.ones(x_len) * C]))
    A = cvxopt.matrix(np.reshape(y, (1, -1)))
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x']).flatten() #30
    # Find support_vectors
    idx = alpha > 1e-6
    SV = x[idx,:]
    # Find w
    w = np.sum(np.array(sol['x'] * y * x), axis=0).reshape(-1)
    # Find b
    alpha_y = np.reshape(alpha[idx], (-1, 1)) * np.reshape(y[idx], (-1, 1))
    b =  np.sum(y[idx]) - np.sum(alpha_y * gaussian_kernel(SV, SV, sigma))
    b /= len(SV)
    cache = {"alpha_y": alpha_y,"SV": SV} 
    return alpha,w,b,cache

def predict(x,cache ,sigma,b):
    alpha_y = cache['alpha_y'] 
    SV = cache['SV'] 
    pred = np.sum(alpha_y * gaussian_kernel(SV, x, sigma), axis=0) + b
    pred_sign = np.sign(pred)
    
    return pred_sign

if __name__ == '__main__':
    data = pd.read_csv("SVM_train.csv",header=None).to_numpy() 
    test = pd.read_csv("SVM_test.csv",header=None).to_numpy() 
    Input=data[:,0:2] #30x2
    Target=data[:,2] 
    sigma=1
    C=1
    alpha,w,b,cache = SVM(Input,Target,sigma,C)
    pred_sign = predict(test, cache, sigma,b)
    print('pred_result=\n{}'.format(np.expand_dims(pred_sign,axis=1)))
    '''
    class_1=np.nonzero(Target+1)
    class_2=np.nonzero(Target-1)
    C1=data[class_1]
    C2=data[class_2]
    pre_1=np.nonzero(pred_sign+1)
    pre_2=np.nonzero(pred_sign-1)
    p1=test[pre_1]
    p2=test[pre_2]
    plt.scatter(C1[:,0],C1[:,1],edgecolors='b')
    plt.scatter(C2[:,0],C2[:,1],edgecolors='r')
    #plt.scatter(p1[:,0],p1[:,1],edgecolors='g')
    #plt.scatter(p2[:,0],p2[:,1],edgecolors='y')
    '''
