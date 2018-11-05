import cv2
import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(sum([ord(c) for c in UBIT]))
COLORS = ["red","green","blue"]
def plot_points(points,clusters,marker):
    for i,x in enumerate(points):
        plt.scatter(x[0], x[1], edgecolor=COLORS[clusters[i]-1], facecolor='white', linewidth='1', marker=marker)
        plt.text(x[0]+0.03, x[1]-0.05, "("+str(x[0])+","+str(x[1])+")", color=COLORS[clusters[i]-1], fontsize='small')

def plot_centroids(centroids,colors,marker):
    for c,color in zip(centroids,colors):
        plt.scatter(c[0], c[1], marker=marker, s=200, c=color)
        plt.text(c[0]+0.03, c[1]-0.05, "("+str(c[0])+","+str(c[1])+")", color=color, fontsize='small')

def dist(points,dims):
    dist = 0
    if(dims == 2):
        dist = math.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2)
    elif(dims == 3):
        dist = math.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2 + (points[0][2]-points[1][2])**2)
    return dist

def make_random_points(dims,num_clusters,type):
    result = []
    for i in range(0,num_clusters): 
        point = []
        for i in range(0,dims):
            point.append( random.randint(1,255))
        result.append(point)
    return result

def computer_kmeans(points,num_clusters,iterns,centeroids = None):
    dims = len(points[0])
    output = []
    mask = []
    temp_output = []
    temp_mask = []
    flag = True
    for i in range(0, num_clusters):
        output.append([])
    if centeroids is None:
        centeroids = make_random_points(dims,num_clusters,np.uint8)
    
    i = 0
    while((iterns > 0 and i < iterns) or(iterns == 0 and flag is True)):
        temp_output = []
        temp_mask = []
        flag = False
        i = i+1
        for j in range(0, num_clusters):
            temp_output.append([])
        for j,point in tqdm(enumerate(points)):
            old_dist = sys.float_info.max
            index = 0
            for k,centeroid in enumerate(centeroids):
                curr_dist = dist((point,centeroid),dims)
                
                if curr_dist < old_dist:
                    index = k
                    old_dist = curr_dist
            temp_output[index].append(points[j])
            temp_mask.append(index+1)
        print(temp_mask)
        plot_points(points,temp_mask,'^')
        plt.savefig("output/task3/task_3_iter"+ str(i)+"_a"".jpg")
        plt.clf()

        
        for j in range(0,num_clusters):
            avg = [float(sum(col))/len(col) for col in zip(*temp_output[j])]
            print(centeroids[j],"avg = ",avg)
            if centeroids[j] != avg:
                flag = True
                centeroids[j] = avg
        plot_centroids(centeroids,COLORS,'.')
        plot_points(points,temp_mask,'^')
        plt.savefig("output/task3/task_3_iter"+ str(i)+"_b"".jpg")
        plt.clf()
    output = temp_output
    mask = temp_mask
   
    return output,centeroids,mask

points= [[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.7,3.4],[6.7,3.1],[5.1,3.8],[6.0,3.0]]
centeroids = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
output,centeroids,mask = computer_kmeans(points,3,2,centeroids)

print(output)
print(centeroids)
print(mask)

