UBIT = 'hparthas'
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
np.random.seed(sum([ord(c) for c in UBIT]))
# read image 1 and convert to BW
m1_clr = cv2.imread('data/tsucuba_left.png')
image1_bw= cv2.cvtColor(m1_clr,cv2.COLOR_BGR2GRAY)

# read image 2 and convert to BW
m2_clr = cv2.imread('data/tsucuba_right.png')
image2_bw = cv2.cvtColor(m2_clr,cv2.COLOR_BGR2GRAY)

# Extract Sift features and compute Descriptors for image 1 and image 2
sift = cv2.xfeatures2d.SIFT_create()
keypoints_mountain1 ,m1_des= sift.detectAndCompute(image1_bw,None)
image1_withkp = cv2.drawKeypoints(m1_clr,keypoints_mountain1,None)
cv2.imwrite('output/task2/task2_sift1.jpg',image1_withkp)

keypoints_mountain2,m2_des = sift.detectAndCompute(image2_bw,None)
image2_withkp =  cv2.drawKeypoints(m2_clr,keypoints_mountain2,None)
cv2.imwrite('output/task2/task2_sift2.jpg',image2_withkp)

def drawlines(img1,img2,lines,pts1,pts2,color):
    r,c = (cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)).shape
    i = 0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color[i],1)
        img1 = cv2.circle(img2,tuple(pt1),5,color[i],-1)
        i = i+1
    return img1


pts1 = []
pts2 = []

bf = cv2.BFMatcher()
matches = bf.knnMatch(m1_des,m2_des, k=2)

for i,(m,n) in enumerate(matches):
        pts2.append(keypoints_mountain2[m.trainIdx].pt)
        pts1.append(keypoints_mountain1[m.queryIdx].pt)

fundamentalmat, mask = cv2.findFundamentalMat(np.array(pts1),np.array(pts2),cv2.FM_RANSAC)
print(fundamentalmat)


pts1 = np.array(pts1)[mask.ravel() == 1]
pts2 = np.array(pts2)[mask.ravel() == 1]

random_points = np.random.randint(0, len(pts1), 10)

selected_point1,selected_point2 = list(), list()
for i, (p1, p2) in enumerate(zip(pts1, pts1)):
    if i in random_points:
        selected_point1.append(p1)
        selected_point2.append(p2)
selected_point1 = np.float32(selected_point1)
selected_point2 = np.float32(selected_point2)
colors = []
for i in range(0,10):
    colors.append(tuple(np.random.randint(0,255,3).tolist()))

img1_lines = cv2.computeCorrespondEpilines(selected_point1.reshape(-1, 1, 2), 2, fundamentalmat)
img1_lines = img1_lines.reshape(-1, 3)
img1_lines1 = drawlines(m1_clr,m2_clr,img1_lines,selected_point1,selected_point2,colors)

img2_lines = cv2.computeCorrespondEpilines(selected_point2.reshape(-1, 1, 2), 2, fundamentalmat)
img2_lines = img1_lines.reshape(-1, 3)
img2_lines1 = drawlines(m2_clr,m1_clr,img2_lines,selected_point2,selected_point1,colors)


stereo = cv2.StereoBM_create(96, blockSize=17)
stereo.setMinDisparity(16)
stereo.setDisp12MaxDiff(0)
stereo.setUniquenessRatio(10)
stereo.setSpeckleRange(32)
stereo.setSpeckleWindowSize(100)
disparity_map = stereo.compute(image1_bw, image2_bw).astype(np.float32) / 16.0
disp_map = (disparity_map - 16)/96

# printing out all the output
plt.imsave('output/task2/task2_disparity.jpg', disp_map, cmap=plt.cm.gray)
cv2.imwrite('output/task2/task2_epi_right.jpg', img2_lines1)
cv2.imwrite('output/task2/task2_epi_left.jpg', img1_lines1)
cv2.imwrite("output/task2/merged.jpg", np.hstack([img2_lines1, img1_lines1]))
