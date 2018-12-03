import numpy as np
import math
import cv2
from tqdm import tqdm


def apply_convolution(image,kernel,size=3):
    rows,cols = image.shape
    new_image = np.zeros((rows,cols), dtype="float32")
    for i in tqdm(range(0,rows - 3)):
        for j in range(0,cols - 3):
            total = 0
            for k in range(0,3):
                for m in range(0,3):
                    total = total + kernel[k,m] * image[i +k,j+m]
            new_image[i + 1,j + 1] = total

    return new_image

#function for task 3
def find_gradient(image1,image2):
    rows,cols = image1.shape[0],image1.shape[1]
    for i in range(rows):
        for j in range(cols):
            image1[i,j] = math.sqrt(image1[i,j]**2 + image2[i,j]**2)
    return image1

def apply_sobel(image):
        sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],dtype="float32")
        sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],dtype="float32")
     
        print("Computing sobel x")
        sobelx_image = apply_convolution(image.copy(),sobelX)
        print("Computing sobel y")
        sobely_image = apply_convolution(image.copy(),sobelY)

        gradient_image = find_gradient(sobelx_image,sobely_image)
        cv2.imwrite("gradient_image.jpg",gradient_image)
        return cv2.imread("gradient_image.jpg",0)

def convert_degree_to_rad(degree):
    return degree * math.pi/180

def get_cos_sin_arrays(input_array):
    for i in range(0,len(input_array)):
        input_array[i] = convert_degree_to_rad(input_array[i])
    cos_arr = input_array.copy()
    for i in range(0,len(input_array)):
        cos_arr[i] = math.cos(cos_arr[i])
        input_array[i] = math.sin(input_array[i])
    return (input_array,cos_arr)


def hough_lines(input_image,sin_arr,cos_arr):
    l,w = input_image.shape
    diagonal = math.ceil(math.sqrt((l)**2 + w**2))
    accumulator = np.zeros((diagonal*2,len(sin_arr)))
    print("Computing Accumulator value")
    for i in tqdm(range(l)):
        for j in range(w):
            if(input_image[i,j]> 150):
                for index in range(len(sin_arr)):
                    degree = math.floor(i  * cos_arr[index] +  j * sin_arr[index] + diagonal)
                    accumulator[degree][index] = accumulator[degree][index] + 1
    return accumulator

def hough_circles(input_image,radius):
    l,w = input_image.shape 
    accumulator = np.zeros((l,w),dtype='int16')
    angles_cos = np.zeros(360,dtype='int16')
    angles_sin = np.zeros(360,dtype='int16')
    
    print("Computing Accumulator value")
    for i in tqdm(range(l)):
        for j in range(w):
            if(input_image[i,j]> 100):
                for angle in range(0,360,2):
                    a = i -  int(radius * math.cos(convert_degree_to_rad(angle)))
                    b = j -  int(radius * math.sin(convert_degree_to_rad(angle)))
                    if a >=0 and a < l and b >=0 and b < w:
                        accumulator[a][b] += 1
    return accumulator

def find_peaks(accumulator,sin_arr,cos_arr,accumulator_threshold,angle_peak_range,rho_peak_range):
    answer_array = []
    rho_range,angle_range = accumulator.shape
    for i in tqdm(range (rho_range)):
        for j in range(angle_range):

            if accumulator[i][j] > accumulator_threshold:
                flag = False
                for k in range (i-rho_peak_range ,i+rho_peak_range):
                    if k >= 0 and k < rho_range :
                        for m in range(j-angle_peak_range ,j+angle_peak_range):
                            if(m >=0 and m< angle_range and accumulator[k][m] > accumulator[i][j] ):
                                flag = True
                                break
                    if flag:
                        break
                if flag == False:
                    if i - rho_range/2 == 0 or sin_arr[j]==0:
                        break
                    x1 = (int)((1  - 1 *  cos_arr[j]/ (i - rho_range/2)) *(i - rho_range/2)  / sin_arr[j])
                    x2 = (int)((1  - 1000 *  cos_arr[j]/ (i - rho_range/2)) *(i - rho_range/2)  / sin_arr[j])
                    answer_array.append(((x1,0),(x2,1000)))
    return answer_array

def draw_lines(image,line_array):
    for i in range(0,len(line_array)):
        cv2.line(image,line_array[i][0],line_array[i][1],(0,255,0),2)
    return image 


def main():

    print("Performing task 3")
    input_image = cv2.imread("input_data/hough.jpg",0)
    output = cv2.imread("input_data/hough.jpg")
    print("Computing sobel")
    sobel_img = apply_sobel(input_image)
    sobel_copy = sobel_img.copy()

    #morphological operations are done inorder to close gaps in lines
    #closing
    kernel = np.ones((3,3), np.uint8)
    sobel_img =cv2.dilate(sobel_img,kernel,1)
    sobel_img= cv2.erode(sobel_img,kernel,1)
    #erosion
    sobel_img= cv2.erode(sobel_img,kernel,1)


    # detection of diagonal lines
    print("Diagonal line detection")
    sin_arr,cos_arr = get_cos_sin_arrays(np.arange(-60.0,-50.0,1))
    accumulator = hough_lines(sobel_img,sin_arr,cos_arr)
    output_lines = find_peaks(accumulator,sin_arr,cos_arr,40,8,50)
    diagonal_output =  draw_lines(output.copy(),output_lines)
    diagonal_lines = len(output_lines)
    cv2.imwrite("output_images/task3/blue_line.jpg",diagonal_output)
    
    #detection of vertical lines
    print("Vertical line detection")
    sin_arr,cos_arr = get_cos_sin_arrays(np.arange(-90.0,-80.0,1))
    accumulator = hough_lines(sobel_img,sin_arr,cos_arr)
    output_lines = find_peaks(accumulator,sin_arr,cos_arr,300,8,30)
    vertical_output =  draw_lines(output.copy(),output_lines)
    vertical_lines = len(output_lines)
    cv2.imwrite("output_images/task3/red_line.jpg",vertical_output)

    print("Circle detection")
    l,w = sobel_copy.shape
    accumulator = hough_circles(sobel_copy,22)
    circle_count = 0

    # Peak finding implementation
    for i in range(l): 
        for j in range(w): 
            if(i > 0 and j > 0 and i < l-1 and j < w-1 and accumulator[i][j] >= 90):
                flag = False
                for k in range (i -30,i+30):
                    if k >= 0 and k < l -1 :
                        for m in range(j - 30,j+30):
                            if(m >=0 and m<w-1 and accumulator[k][m] > accumulator[i][j]):
                                flag = True
                                break
                    if flag:
                        break
                if flag == False:
                    cv2.circle(output,(j,i),22,(0,255,0),3)
                    circle_count += 1
    cv2.imwrite("output_images/task3/coin.jpg",output)

    print("Detected Vertical lines = ",vertical_lines)
    print("Detected Diagonal lines = ",diagonal_lines)
    print("Detected Circles = ",circle_count)
    
    print("complete execution")

if __name__ == "__main__":
    main()
