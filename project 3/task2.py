import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def apply_convolution(image,kernel,size=3):
    rows,cols = image.shape
    new_image = np.zeros((rows,cols), np.float)
    max_val =0
    for i in tqdm(range(0,rows-size)):
        for j in range(0,cols-size):
            total = 0
            for k in range(0,size):
                for m in range(0,size):
                    total = total + kernel[k,m] * image[i +k,j+m]
            new_image[int(i+size/2+1),int(j+size/2+1)] = abs(total)
            if(max_val < abs(total)):
                max_val = abs(total)
    return new_image,max_val

print("performing part 1")
input_image = cv2.imread("input_data/point.jpg",0)
KERNEL = np.array([[-1,-1,-1,-1,-1],
                   [-1,-1,-1,-1,-1],
                   [-1,-1,24,-1,-1],
                   [-1,-1,-1,-1,-1],
                   [-1,-1,-1,-1,-1]])


output_image,max_val = apply_convolution(input_image,KERNEL,5)
l,w = input_image.shape
threshold = (0.9 * max_val)
for i in range(0,l):
    for j in range(0,w):
        if output_image[i][j] > threshold:
            output_image[i][j] = 255
            x1,y1 = j-1,i-1
        else :
            output_image[i][j] = 0 

cv2.circle(output_image, (x1,y1),10, (255,255,255), thickness=2, lineType=8, shift=0)
cv2.imwrite("output_images/task2/res_point1.jpg",output_image)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output_image,"("+str(x1)+","+str(y1)+")",(x1-10, y1+40), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
cv2.imwrite("output_images/task2/point_plotted.jpg",output_image)

print("performing part 2")
input_image = cv2.imread("input_data/segment.jpg",0)
l,w = input_image.shape
histogram = np.zeros(256)
for i in range(0,l):
    for j in range(0,w):
        if(input_image[i][j] > 0):
            histogram[input_image[i][j]] += 1
plt.plot(histogram)
plt.title('Image Histogram')
plt.grid(True)
plt.savefig('output_images/task2/res_segment_histogram.png')
for i in range(0,l):
    for j in range(0,w):
        if(input_image[i][j] < 205):
            input_image[i][j] = 0

points = [[(160,125),(200,165)],[(246,77),(300,204)],[(329,24),(365,287)],[(387,41),(423,251)]]
for point in points:
    cv2.rectangle(input_image,point[0],point[1],(255,255,255),2)
cv2.imwrite("output_images/task2/res_segment_image.jpg",input_image)
print("Execution complete")
