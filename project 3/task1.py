import cv2;

def apply_erosion(image):
    image_temp = image.copy()
    l,w = image.shape
    for i in range(1,l-1):
        for j in range(1,w-1):
            if image_temp[i -1][j -1] == 0 or image_temp[i-1][j] == 0 or image_temp[i -1][j +1] == 0 or image_temp[i ][j-1] == 0 or image_temp[i][j] == 0 or image_temp[i][j+1] == 0 or image_temp[i+1][j-1] == 0 or image_temp[i+1][j] == 0 or image_temp[i+1][j+1] == 0:
                    image[i][j] =0 
    return image
                    

def apply_dilation(image):
    image_temp = image.copy()
    l,w = image.shape
    for i in range(1,l-1):
        for j in range(1,w-1):
            if image_temp[i -1][j -1] == 255 or image_temp[i-1][j] == 255 or image_temp[i -1][j +1] == 255 or image_temp[i ][j-1] == 255 or image_temp[i][j] == 255 or image_temp[i][j+1] == 255 or image_temp[i+1][j-1] == 255 or image_temp[i+1][j] == 255 or image_temp[i+1][j+1] == 255:
                    image[i][j] =255 
    return image
                    
def apply_opening(image):
    output = apply_erosion(image)
    output = apply_dilation(output)
    return output

def apply_closing(image):
    output = apply_dilation(image)
    output = apply_erosion(output)
    return output

# Read input and compute sobel image 
print("performing task 1")

# Algorithm 1
image = cv2.imread("input_data/noise.jpg",cv2.IMREAD_GRAYSCALE)
image_1 = apply_opening(image.copy())
cv2.imwrite("output_images/task1/algo1_opening.jpg",image_1)

image_1 = apply_closing(image_1)
cv2.imwrite("output_images/task1/res_noise1.jpg",image_1)
algo1_output = image_1.copy()

eroded_noise1 = apply_erosion(image_1.copy())
boundary_noise1 = image_1 - eroded_noise1
cv2.imwrite("output_images/task1/res_bound1.jpg",boundary_noise1)

#Algorithm 2
image_2 = apply_closing(image.copy())
cv2.imwrite("output_images/task1/algo2_opening.jpg",image_2)
image_2 = apply_opening(image_2)
cv2.imwrite("output_images/task1/res_noise2.jpg",image_2)
algo2_output = image_2.copy()
eroded_noise2 = apply_erosion(image_2.copy())
boundary_noise2 = image_2 - eroded_noise2
cv2.imwrite("output_images/task1/res_bound2.jpg",boundary_noise2)

# Computing difference between Algorithm 1 and Algorithm 2
cv2.imwrite("output_images/task1/difference.jpg", algo2_output - algo1_output)

print("Execution complete")
