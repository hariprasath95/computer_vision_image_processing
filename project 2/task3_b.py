UBIT = 'hparthas'
import cv2
import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(sum([ord(c) for c in UBIT]))
COLORS = ["red","green","blue"]
def plot_points(points,clusters,marker,text=True):
    for i,x in enumerate(points):
        plt.scatter(x[0], x[1], edgecolor=COLORS[clusters[i]], facecolor='white', linewidth='1', marker=marker)
        if text:
            plt.text(x[0]+0.03, x[1]-0.05, "("+str(x[0])+","+str(x[1])+")", color=COLORS[clusters[i]], fontsize='small')

#Plot the centroids and its values if text is true
def plot_centroids(centroids,colors,marker,text=True):
    for i,c in enumerate(centroids):
        plt.scatter(c[0], c[1], marker=marker, s=200, c=COLORS[i])
        if text:
            plt.text(c[0]+0.03, c[1]-0.05, "("+str(c[0])+","+str(c[1])+")", color=COLORS[i], fontsize='small')
def make_image(kmeans_input,mask,centeroids,h,w):
    for i in range(0,len(mask)):
        kmeans_input[i] = centeroids[mask[i]]
    return kmeans_input.reshape((h, w, 3))

def make_random_points(points,num_clusters):
    centeroids = points.copy()
    np.random.shuffle(centeroids)
    return centeroids[:num_clusters]

def computer_kmeans(points,num_clusters,iterns,centeroids = None):
    flag = True
    if centeroids is None:
        centeroids = make_random_points(points,num_clusters)
    
    i = 0
    while((iterns > 0 and i < iterns) or(iterns == 0 and flag is True)):
        flag = False
        i = i+1
        # reference https://flothesof.github.io/k-means-numpy.html
        distances = np.sqrt(((np.array(points) - np.array(centeroids)[:, np.newaxis])**2).sum(axis=2))
        min_values = np.argmin(distances, axis=0)
        new_centeroids = np.array([points[min_values==k].mean(axis=0) for k in range(np.array(centeroids).shape[0])],dtype = int)
        
        if np.array_equiv(new_centeroids,np.array(centeroids)):
            print("reached_convergence after", i,"iterations")
            break
        else:
            flag = True
        centeroids = new_centeroids
    return centeroids, min_values

baboon_image = cv2.imread("data/baboon.jpg")
(h,w) = baboon_image.shape[:2]

kmeans_input = baboon_image.reshape((baboon_image.shape[0]* baboon_image.shape[1],3))

centeroids,mask = computer_kmeans(kmeans_input.copy(),3,0)
image = make_image(kmeans_input.copy(),mask,centeroids,h,w)
cv2.imwrite("output/task3/task3_baboon_3.jpg",image)

centeroids,mask = computer_kmeans(kmeans_input.copy(),5,0)
image = make_image(kmeans_input.copy(),mask,centeroids,h,w)
cv2.imwrite("output/task3/task3_baboon_5.jpg",image)

centeroids,mask = computer_kmeans(kmeans_input.copy(),10,0)
image = make_image(kmeans_input.copy(),mask,centeroids,h,w)
cv2.imwrite("output/task3/task3_baboon_10.jpg",image)

centeroids,mask = computer_kmeans(kmeans_input.copy(),20,0)
image = make_image(kmeans_input.copy(),mask,centeroids,h,w)
cv2.imwrite("output/task3/task3_baboon_20.jpg",image)











