from lib import basics
import numpy as np
import cv2
import math
rgb_image = cv2.imread("task2.jpg")
original_image = cv2.imread("task2.jpg",0)

img_scaled = np.empty(shape=5,dtype=object)
img_scaled[0] = original_image.copy()
img_scaled[1] =  np.array(basics.scale_half(img_scaled[0])).copy()
img_scaled[2] =  np.array(basics.scale_half(img_scaled[1])).copy()
img_scaled[3] =  np.array(basics.scale_half(img_scaled[2])).copy()


gaussian_matrix = [[1/math.sqrt(2),1,math.sqrt(2),2,2*math.sqrt(2)],
                   [math.sqrt(2),2,2*math.sqrt(2),4,4*math.sqrt(2)],
                   [2*math.sqrt(2),4,4*math.sqrt(2),8,8*math.sqrt(2)],
                   [4*math.sqrt(2),8,8*math.sqrt(2),16,16*math.sqrt(2)]
                   ]
image_array = np.empty((4,5), dtype = object)
DoG_array = np.empty((4,4),dtype = object)
for i in range(0,4):
    for j in range(0,5):
        gaussian_kernel = basics.calculate_gaussian(gaussian_matrix[i][j],7)
        image_array[i,j] = basics.apply_convolution(basics.add_padding(img_scaled[i],7//2),gaussian_kernel,7)
        if j > 0:
            DoG_array[i,j-1] = basics.computer_difference(image_array[i,j],image_array[i,j-1]) 
        print(gaussian_kernel)


key_point_array = np.empty((4,2),dtype = object)
key_point_array_1 = basics.find_keys(DoG_array[0][0],DoG_array[0][1],DoG_array[0][2])
key_point_array_2 = basics.find_keys(DoG_array[0][1],DoG_array[0][2],DoG_array[0][3])
key_point_array_21 = basics.find_keys(DoG_array[1][0],DoG_array[1][1],DoG_array[1][2])
key_point_array_22 = basics.find_keys(DoG_array[1][1],DoG_array[1][2],DoG_array[1][3])
key_point_array_31 = basics.find_keys(DoG_array[2][0],DoG_array[2][1],DoG_array[2][2])
key_point_array_32 = basics.find_keys(DoG_array[2][1],DoG_array[2][2],DoG_array[2][3])
key_point_array_41 = basics.find_keys(DoG_array[3][0],DoG_array[3][1],DoG_array[3][2])
key_point_array_42 = basics.find_keys(DoG_array[3][1],DoG_array[3][2],DoG_array[3][3])
    
plot_image_1 = rgb_image.copy()
img_copy_1 =img_scaled[0].copy()
img_copy_2 =np.array(img_scaled[1]).copy()
img_copy_3 =np.array(img_scaled[2]).copy()
img_copy_4 =np.array(img_scaled[3]).copy()

for point in key_point_array_1:
        cv2.circle(img_scaled[0],point,1,(0,255,0))
for point in key_point_array_2:
       cv2.circle(img_copy_1,point,1,(255,0,0))

for point in key_point_array_21:
        cv2.circle(img_scaled[1],point,1,(0,255,0))
for point in key_point_array_22:
       cv2.circle(img_copy_2,point,1,(255,0,0))

for point in key_point_array_31:
        cv2.circle(img_scaled[2],point,1,(0,255,0))
for point in key_point_array_32:
       cv2.circle(img_copy_3,point,1,(255,0,0))

for point in key_point_array_41:
        cv2.circle(img_scaled[3],point,1,(0,255,0))
for point in key_point_array_42:
       cv2.circle(img_copy_4,point,1,(255,0,0))


cv2.imshow('image1',img_scaled[0])

cv2.imshow('image12',img_copy_1)

cv2.imshow('image21',img_scaled[1])

cv2.imshow('image22',img_copy_2)

cv2.imshow('image31',img_scaled[2])

cv2.imshow('image32',img_copy_3)

cv2.imshow('image41',img_scaled[3])

cv2.imshow('image42',img_copy_4)

cv2.waitKey(0)


