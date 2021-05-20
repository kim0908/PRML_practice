# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:42:14 2021

@author: CSDSP_KIM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sign(z):
    if z >= 0:
        return 1
    else:
        return -1
    
if __name__ == '__main__':
    PLA_data = pd.read_csv("PLA_data.csv",header=None,names=['phi_1', 'phi_2','target'])
    w = np.array([0.,0.,0.])
    error = 1
    step = 0
    while error != 0:
        error = 0
        for i in range(len(PLA_data)):
            x,y = np.concatenate((np.array([1.]), np.array(PLA_data.iloc[i])[:2])), np.array(PLA_data.iloc[i])[2]
            if sign(np.dot(w,x)) != y:
                print("step: "+str(step+1))
                step += 1
                error += 1
                w += y*x            
                print("x: " + str(x))            
                print("w: " + str(w))
                
                plt.figure()
                x_decision_boundary = np.linspace(-2,2)
                y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
                sns.scatterplot(x='phi_1',y='phi_2',data=PLA_data,  hue ='target')
                plt.plot(x_decision_boundary, y_decision_boundary,'r')
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)