def apply_convolution(image,kernel,size=3):
    rows,cols = image.shape
    new_image = np.zeros((rows,cols), np.uint8)
    for i in range(0,rows-size):
        for j in range(0,cols-size):
            total = 0
            for k in range(0,size):
                for m in range(0,size):
                    total = total + kernel[k,m] * image[i +k,j+m]
            new_image[i+size/2+1,j+size/2+1] = abs(total)/(size * size)

    return new_image
def find_gradient(image1,image2):
    rows,cols = image1.shape[0],image1.shape[1]
    for i in range(rows):
        for j in range(cols):
            image1[i,j] = math.sqrt(image1[i,j]**2 + image2[i,j]**2)
    return image1

import cv2
import numpy as np
import math
img = cv2.imread("task1.png",0)
blur_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
kernel_v = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kernel_h = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

# blurred_image = apply_convolution(img,blur_kernel)
edge_detected_image_v = apply_convolution(img,kernel_v)
edge_detected_image_h = apply_convolution(img,kernel_h)

cv2.imshow('img',img)
# cv2.imshow('blurred_image',blurred_image)
cv2.imshow('edge_detected_image_v',edge_detected_image_v)
cv2.imshow('edge_detected_image_h',edge_detected_image_h)
cv2.imshow('gradient',find_gradient(edge_detected_image_v,edge_detected_image_h))
cv2.waitKey()


