import numpy as np
import cv2
import math
def apply_convolution(image,kernel,size=3):
    rows,cols = image.shape
    new_image = np.zeros((rows,cols), dtype='float32')
    sum_total =0.0
    for i in range(0,size):
        for j in range(0,size):
          sum_total = sum_total + kernel[i][j]  
    for i in range(0,rows-size):
        for j in range(0,cols-size):
            total = 0.0
            for k in range(0,size):
                for m in range(0,size):
                    total = total + kernel[k,m] * image[i +k,j+m]
            new_image[i+math.floor(size/2)+1,j+math.floor(size/2)+1] = abs((total)/(sum_total))
    return new_image
def apply_convolution1(image,kernel,size=3):
    rows,cols = image.shape
    new_image = np.zeros((rows,cols), dtype='uint8')
    for i in range(0,3):
        for j in range(0,3):
            total = 0
            for k in range(0,size):
                for m in range(0,size):
                    total = total + kernel[k,m] * image[i +k,j+m]
            new_image[i+math.floor(size/2)+1,j+math.floor(size/2)+1] = total

    return new_image
def find_gradient(image1,image2):
    rows,cols = image1.shape[0],image1.shape[1]
    for i in range(rows):
        for j in range(cols):
            image1[i,j] = math.sqrt(image1[i,j]**2 + image2[i,j]**2)
    return image1

def scale_half(image):
    scaled_x = image[::2]
    scaled_xy = scaled_x[:,1::2]
    return scaled_xy

def calculate_gaussian(sigma_value,size):
    print(size)
    sum = 0.0
    print(sigma_value)
    new_kernel = np.zeros((size,size),dtype = float)
    for i in range(int(-math.floor(size/2)),int(math.floor(size/2)+1)):
        for j in range(int(-math.floor(size/2)),int(math.floor(size/2)+1)):
            
            numerator =  math.exp(-float((i*i+j*j))/float(2*sigma_value*sigma_value))
            denominator = (2*(math.pi)* math.pow(sigma_value,2))            
            new_kernel[(i+math.floor(size/2)),(j+math.floor(size/2))] = numerator/denominator            
            sum = sum + new_kernel[i+math.floor(size/2),j+math.floor(size/2)]

    return new_kernel

def computer_difference(image1,image2):
    rows,cols = image1.shape
    new_image = np.zeros((rows,cols),dtype = np.float)
    for i in range(0,rows):
        for j in range (0 , cols):
            new_image[i,j] = abs(int(image1[i,j])-int(image2[i,j]))
    return new_image

def find_keys(d1,d2,d3):
    keys = []
    for i in range(1,d1.shape[0]-1):
        for j in range(1,d1.shape[1]-1):
            d11 = d1[i-1:i+2,j-1:j+2] #3X3 matrix in dog1
            d22 = d2[i-1:i+2,j-1:j+2] #3X3 matrix in dog2
            d33 = d3[i-1:i+2,j-1:j+2] #3X3 matrix in dog3
            center = d22[1,1] #Center element in dog2 - d22
            d22 = np.delete(d22[1], 1) #Delete center element from the dog2 - d22
            mn1,mx1 = d11.min(),d11.max() #Min and Max element in d11
            mn2,mx2 = d22.min(),d22.max() #Min and Max element in d22
            mn3,mx3 = d33.min(),d33.max() #Min and Max element in d33
            mins = min([mn1,mn2,mn3]) #Mins between the 3 mins above
            maxs = max([mx1,mx2,mx3]) #Maxs between the 3 maxs above
            #Keypoint should be either less than the min or greater than the max
            if((center<mins) or (center>maxs)) : 
                keys.append((j,i))
    return keys
def add_padding(img, padding):
    img = np.pad(img, (padding, padding), mode='constant', constant_values=0)
    return img