#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
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

def plot1(train_x, train_y, kmeans_results, kmeans_metrics,params,result_dir):
    plt.figure(figsize=(8, 10))
    plt.subplot(3, 2, 1)
    plt.title('Samples')
    plt.scatter(train_x[:,0], train_x[:,1],c=train_y)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    subplot_counter = 1
    plt.subplot(3, 2, subplot_counter)
    for i in range(len(params)):
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        for j, l in enumerate(kmeans_results[i]):
            plt.plot(train_x[:,0][j], train_x[:,1][j], color=colors[l],marker=markers[l],ls='None')
        plt.title('K = %s, silhouette score = %.03f' % (params[i],kmeans_metrics[i]))
        #plt.xlabel('x')
        #plt.ylabel('x')
        #plt.show()
    plt.savefig(result_dir+"fig1.png")

def plot2(train_x, train_y, kmeans_results, kmeans_metrics,params,result_dir):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    ax.scatter(train_x[:,0], train_x[:,1], train_x[:,2],c=train_y)
    ax.set_title('Samples')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    subplot_counter = 1
    for i in range(len(params)):
        subplot_counter += 1
        ax = fig.add_subplot(3, 2, subplot_counter, projection='3d')
        for j, l in enumerate(kmeans_results[i]):
            ax.scatter(train_x[:,0][j], train_x[:,1][j], train_x[:,2][j], color=colors[l],marker=markers[l])
        ax.set_title('K = %s, silhouette score = %.03f' % (params[i],kmeans_metrics[i]))
    plt.savefig(result_dir+"fig2.png")

if __name__ == '__main__':
    # step 1: read data files
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.targe
    data_file = "../data/iris.txt"
    result_dir = "../result/"
    train_x, train_y, train_label, varName,speciesName = read_data(data_file)

    # step 2: data normalization


    # step 3: modelling: k-means clustering
    params = [2, 3, 4, 5, 8]
    kmeans_results = []
    kmeans_metrics = []
    for n in params:
        np.random.seed(5)
        kmeans_model = KMeans(n_clusters=n).fit(train_x)
        #centroids = kmeans_model.cluster_centers_
        kmeans_results.append(kmeans_model.labels_)
        kmeans_metrics.append(metrics.silhouette_score(train_x,kmeans_model.labels_,metric='euclidean'))

    ## # step 4: graphical visualization
    plot1(train_x, train_y, kmeans_results, kmeans_metrics,params,result_dir)
    plot2(train_x, train_y, kmeans_results, kmeans_metrics,params,result_dir)
