# -*- coding: utf-8 -*-
"""
Created on Sun May 22 23:47:34 2021

@author: CSDSP_KIM
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def euclid_dis(A,B):
    dis = np.zeros([len(A),len(B)])
    for b in range(len(B)):
        for a in range(len(A)):
            dis[a,b] = np.sqrt(sum(np.power((A[a] - B[b]), 2)))
    return dis
def gaussian(x, mu, cov):
    result=np.zeros((len(x),1))
    for n in range(len(x)):
        try:
            part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
            part2 = (-1/2) * ((x[n]-mu).T.dot(np.linalg.inv(cov))).dot((x[n]-mu))
            result[n] = float(part1 * np.exp(part2))
        except:
            cov = np.expand_dims(x[n],axis=0).T.dot(np.expand_dims(x[n],axis=0))
            part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
            part2 = (-1/2) * ((x[n]-mu).T.dot(np.linalg.inv(cov))).dot((x[n]-mu))
            result[n] = float(part1 * np.exp(part2))
    return result.squeeze()

def K_means(K,x):
    loop=1000
    I = np.eye(K)
    center = x[np.random.choice(len(x), K , replace=False)] #len(x) choose K
    for l in range(loop):
        prev_center = np.copy(center)
        D = euclid_dis(x,center)
        index = np.argmin(D, axis=1)
        index = I[index]
        center = np.sum(x[:, None, :] * index[:, :, None], axis=0) / np.sum(index, axis=0)[:, None]
        if np.allclose(prev_center, center):
            break               
    return center , index
        
def E_step(x,mu,cov,pi):
    unnormal_prob_list = []
    for k in range(K):
        mu_tmp, cov_tmp, pi_tmp = mu[k, :], cov[k, :, :], pi[k]
        unnormal_prob = pi_tmp * gaussian(x , mu_tmp, cov_tmp)
        unnormal_prob_list.append(np.expand_dims(unnormal_prob, -1))
    preds = np.concatenate(unnormal_prob_list, axis=1) #list ->array
    log_likelihood = np.sum(preds, axis=1)
    log_likelihood = np.sum(np.log(log_likelihood))
    preds = preds / np.sum(preds, axis=1, keepdims=True)
    prob = np.asarray(preds)
    return prob ,log_likelihood

def M_step(x,prob):
    new_mu_list, new_cov_list, new_pi_list = [], [], []
    count = np.sum(prob, axis=0)
    for k in range(K):
        #update mu
        new_mu = np.sum(np.expand_dims(prob[:, k], -1) * x, axis=0)
        new_mu /= count[k]
        new_mu_list.append(new_mu)
        #update cov
        vector = np.subtract(x, np.expand_dims(new_mu, 0))
        new_cov = np.matmul(np.transpose(np.multiply(np.expand_dims(prob[:, k], -1), vector)), vector)
        new_cov /= count[k]
        new_cov_list.append(new_cov)
        #update pi
        new_pi_list.append(count[k] / np.sum(count))

    mu = np.asarray(new_mu_list)
    cov = np.asarray(new_cov_list)
    pi = np.asarray(new_pi_list)
    return mu,cov,pi

def EM(x,mu_km,index,K):
    # Initial (pi and cov)
    Cov_km, PI =[], []
    N_k = np.sum(index, axis=0)
    for k in range(K):
        tmp = np.subtract(x, np.expand_dims(mu_km[k], 0))
        cov_km = np.matmul(np.transpose(np.multiply(np.expand_dims(index[:, k], -1), tmp)), tmp)
        cov_km /= N_k[k]
        Cov_km.append(cov_km)
        PI.append(N_k[k] / np.sum(N_k))
    #loop
    likely=[]
    mu=np.asarray(mu_km)
    cov=np.asarray(Cov_km)
    pi=np.asarray(PI)
    for _ in range(100): 
        prob ,log_likelihood = E_step(x,mu,cov,pi)
        mu,cov,pi = M_step(x,prob)
        likely.append(log_likelihood)
    return prob,likely ,mu
        
if __name__ == '__main__':
    for i in range(3):
        Image = cv2.imread('image{}.jpg'.format(i+1))
        x_1 = np.reshape(Image,(np.shape(Image)[0]*np.shape(Image)[1],np.shape(Image)[2]))
        K_list=[2,3,5,10,20]
        for K in K_list:
            print('img_{} k={}'.format(i+1,K))
            mu_km,index = K_means(K,x_1)
            print('After kmeans mu=\n{}\n'.format(mu_km))
            prob,likely,mu=EM (x_1,mu_km,index,K)
            print('After EM mu=\n{}\n'.format(mu))
            plt.figure()
            plt.plot(likely)
            plt.savefig('pic/likely_{}_K={}.jpg'.format(i+1,K))
            plt.ioff()
            plt.clf
            #show img
            img_h = np.shape(Image)[0]
            img_w = np.shape(Image)[1]
            map_prob = np.reshape(prob, (img_h, img_w, K))
            out_img = np.zeros((img_h, img_w, 3))
            pixel_color = [(int(mu[k,0]),int(mu[k,1]),int(mu[k,2])) for k in range(K)]
            for h in range(img_h):
                for w in range(img_w):
                    max_prob = np.argmax(map_prob[h, w, :])
                    out_img[h,w,:] = pixel_color[max_prob]
            out_img=out_img.astype(int)[:,:,::-1]
            plt.imshow(out_img)
            plt.savefig('pic/img{}_K={}.jpg'.format(i+1,K))
            plt.clf
