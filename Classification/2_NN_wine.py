# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:11:20 2021

@author: CSDSP_KIM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ReLU(x):
    return x * (x > 0)
def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
    sigmoid=np.zeros((x.shape))
    for n in range(len(x.T)):
        for i in range(len(x)):
            sigmoid[i,n]=1 / (1 + np.exp(-x[i,n]))
    return sigmoid


def softmax(x):
    softmax = []
    for n in range(len(x.T)):
        total = 0
        x_max=np.max(x[:,n])
        for k in range(len(x)):
            total += np.exp(x[k,n]-x_max)
        for k in range(len(x)):
            tmp = (np.exp(x[k,n]-x_max))/total
            if tmp == 0:
                tmp = tmp + 10**(-6)
            softmax.append(tmp)
    softmax = np.reshape(softmax,x.shape)
    return softmax

def layer_sizes(X, Y):
    n_x = X.shape[0] # input 
    n_h = 8 # hidden
    n_y = Y.shape[0] #output
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1)) 
   
    assert (W1.shape == (n_h, n_x))    
    assert (b1.shape == (n_h, 1))    
    assert (W2.shape == (n_y, n_h))    
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1, 
                  "b1": b1,                 
                  "W2": W2,                  
                  "b2": b2}                  
    return parameters

def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']    
    
    A1 = np.dot(W1, X) + b1
    Z1 = sigmoid(A1)
    #Z1 = np.tanh(A1)
    #Z1 = ReLU(A1)
    A2 = np.dot(W2, Z1) + b2
    Z2 = softmax(A2)    
    assert(A2.shape == (b2.shape[0], X.shape[1]))
    cache = {"A1": A1,
             "Z1": Z1,
             "A2": A2,                   
             "Z2": Z2}    
    return Z2, cache

def compute_cost(Z2, Y, parameters):
    cross_ero=0
    for n in range(len(Y.T)):
        for k in range(len(Y)):
            if Z2[k,n]==0:
                cross_ero=cross_ero+Y[k,n]*np.log(Z2[k,n]+10**-6)
            else:
                cross_ero=cross_ero+Y[k,n]*np.log(Z2[k,n])
    loss = -(cross_ero)   
    return loss
    
def Accuracy(Z, target):
    corret=0
    for n in range(len(Z.T)):
        pre = np.argmax(Z[:,n])
        tar = np.argmax(target[:,n])
        if pre == tar:
            corret=corret+1
    return corret/len(Z.T)
             
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]    
    W1 = parameters['W1']
    W2 = parameters['W2']    
    Z1 = cache['Z1']
    Z2 = cache['Z2']    
    dA2 = Z2-Y
    dW2 = 1/m * np.dot(dA2, Z1.T)
    db2 = 1/m * np.sum(dA2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dA2)*(Z1-np.power(Z1, 2))
    #dA1 = np.dot(W2.T, dA2)*(1-np.power(Z1, 2))
    #dA1 = np.dot(W2.T, dA2)*dReLU(Z1)
    dW1 = 1/m * np.dot(dA1, X.T)
    db1 = 1/m * np.sum(dA1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,                      
             "dW2": dW2,             
             "db2": db2}   
    return grads

def update_parameters(parameters, grads, num_data,learning_rate = 0.3):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']    

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']    

    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rate
 
    return parameters

def nn_model(X, Y,X_val, Y_val,n_h, learning_rate, epoch = 200):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    num_data = X.shape[1]
    #Initialize
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #Loop
    Loss_train=[]
    Loss_val=[]
    Acc_train=[]
    Acc_val=[]
    for e in range(epoch):
        Batch_cost=0
        cost=[]
        cost_val=[]
        Batch_acc=0
        acc=[]
        acc_val=[]
        for n in range(num_data):
            x=np.reshape(X[:,n],(-1,1))
            y=np.reshape(Y[:,n],(-1,1))
            # Forward
            Z2, cache = forward_propagation(x, parameters)     
            loss = compute_cost(Z2, y, parameters)
            correct = Accuracy(Z2, y)
            # Back
            grads = backward_propagation(parameters, cache, x, y)        
            # Update
            parameters = update_parameters(parameters, grads, num_data,learning_rate=0.3 )
            cost.append(loss)
            acc.append(correct)
        
        Batch_cost=np.sum(cost)
        Loss_train.append(Batch_cost)
        Batch_acc=np.sum(acc)/num_data
        Acc_train.append(Batch_acc)
        if e % 10 == 0:            
            print ("Train_loss after epoch %i: %f" %(e, Batch_cost))
        #Validation
        for n in range(X_val.shape[1]):
            x_val=np.reshape(X_val[:,n],(-1,1))
            y_val=np.reshape(Y_val[:,n],(-1,1))
            Z2_val, _ = forward_propagation(x_val, parameters)
            loss_val = compute_cost(Z2_val, y_val, parameters)
            cost_val.append(loss_val)
            correct_val = Accuracy(Z2_val, y_val)
            acc_val.append(correct_val)
        Batch_cost_val=np.sum(cost_val)
        Loss_val.append(Batch_cost_val)
        Batch_acc_val=np.sum(acc_val)/X_val.shape[1]
        Acc_val.append(Batch_acc_val)
        if e % 10 == 0:            
            print ("Val_loss after epoch %i: %f" %(e, Batch_cost_val))  
    return parameters,Loss_train,Loss_val,Acc_train,Acc_val

if __name__ == '__main__':
    wine_train = pd.read_csv("wine_train.csv",header=None)
    wine_train=np.array(wine_train.sample(frac=1.0))     
    Data= wine_train[:,3:16]
    Target= wine_train[:,0:3]

    num_samples=48
    Val = Data[0 : num_samples].T
    Val_T = Target[0: num_samples].T
    Train_T= Target[num_samples : -1].T    
    Train= Data[num_samples : -1].T
    #normalize
    tmp=Train[-1,:]
    tmp=(tmp-np.mean(tmp))/np.std(tmp)
    Train[-1,:]=tmp
    
    learning_rate=0.005
    n_h=70
    parameters,Loss_train,Loss_val ,Acc_train,Acc_val =nn_model(Train, Train_T,Val,Val_T, n_h,learning_rate, epoch = 200)
    
    plt.figure()
    plt.plot(Loss_train)
    plt.plot(Loss_val)
    plt.title('wine_Loss lr={} nh={} ReLU'.format(learning_rate,n_h))
    plt.legend(['train', 'validation'])
    plt.show()
    
    plt.figure()
    plt.plot(Acc_train)
    plt.plot(Acc_val)
    plt.title('wine_Accuracy lr={} nh={} ReLU'.format(learning_rate,n_h))
    plt.legend(['train', 'validation'])
    plt.show()
    
    TEST_data = np.array(pd.read_csv("wine_test.csv",header=None)).T
    Z2_test, _ = forward_propagation(TEST_data, parameters)
    predict=np.zeros((len(Z2_test.T),3))
    for n in range(len(Z2_test.T)):
        predict[n,np.argmax(Z2_test[:,n])] = 1.0

    print('Test_prediction=\n{}'.format(predict))    