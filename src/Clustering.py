#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data preparation
Created on 20180516
@author: peishun
"""
import numpy as np
import matplotlib.pyplot as plt
# required for 3D projection to work below
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import metrics 
# from sklearn import datasets


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
def read_data(data_file):  
    '''
    with open('pi_digits.text') as f:
    　　　　lines = f.readlines()
    　　　　print(lines)
    with open('programming.text','w') as f:
    　　　　f.write("I love programming")
    '''
    import csv
    lines=csv.reader(open(data_file),delimiter=" ")
    samples=[]  
    samples.extend(lines)
    varName=samples[0][:-1]
    sepalL=[]
    sepalW=[]
    petalL=[]
    petalW=[]
    species=[]
    for i in samples: 
        if is_number(i[0]):
            sepalL.append(float(i[0])) 
            sepalW.append(float(i[1]))
            petalL.append(float(i[2]))
            petalW.append(float(i[3]))
            species.append(i[4])
    train_x = np.c_[sepalL,sepalW,petalL,petalW]
    # list and array exchange
    #train_x = np.array(list(zip(sepalL[:],sepalW[:],petalL[:],petalW[:])))
    train_label = species
    train_y=[]
    speciesName = list(np.unique(species))
    for i in train_label:
        if i==speciesName[0]:
            train_y.append(0)
        elif i==speciesName[1]:
            train_y.append(1)
        else:
            train_y.append(2)
    return train_x, train_y, train_label, varName,speciesName




fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()



if __name__ == '__main__':
    # step 1: read data files
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.targe
    data_file = "../data/iris.txt"
    train_x, train_y, train_label, varName,speciesName = read_data(data_file)
    # step 2: data normalization


    # step 3: k-means clustering 
    plt.figure(figsize=(8, 10))  
    plt.subplot(3, 2, 1) 
    plt.title('Samples') 
    plt.scatter(train_x[:,0], train_x[:,1],c=train_y) 
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']  
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']  
    tests = [2, 3, 4, 5, 8]  
    subplot_counter = 1  
    for t in tests:  
        subplot_counter += 1  
        plt.subplot(3, 2, subplot_counter) 
        np.random.seed(5)
        kmeans_model = KMeans(n_clusters=t).fit(train_x)
        #centroids = kmeans_model.cluster_centers_ 
        for i, l in enumerate(kmeans_model.labels_): 
            # step 4: graphical visualization
            plt.plot(train_x[:,0][i], train_x[:,1][i], color=colors[l],marker=markers[l],ls='None')   
            plt.title('K = %s, silhouette score = %.03f' % (t,metrics.silhouette_score(train_x,kmeans_model.labels_,metric='euclidean'))) 
     
    ## # step 4: 3D graphical visualization
    subplot_counter = 1  
    for t in tests:  
        subplot_counter += 1  
        plt.subplot(3, 2, subplot_counter) 
        np.random.seed(5)
        kmeans_model = KMeans(n_clusters=t).fit(train_x)
        #centroids = kmeans_model.cluster_centers_ 
        for i, l in enumerate(kmeans_model.labels_): 
            # step 4: graphical visualization
            plt.plot(train_x[:,0][i], train_x[:,1][i], color=colors[l],marker=markers[l],ls='None')   
            plt.title('K = %s, silhouette score = %.03f' % (t,metrics.silhouette_score(train_x,kmeans_model.labels_,metric='euclidean'))) 

