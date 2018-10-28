import cv2
from matplotlib import pyplot as plt

m1_clr = cv2.imread('data/mountain1.jpg')
mountain1= cv2.cvtColor(m1_clr,cv2.COLOR_BGR2GRAY)
m2_clr = cv2.imread('data/mountain2.jpg')
mountain2 = cv2.cvtColor(m2_clr,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints_mountain1 ,m1_des= sift.detectAndCompute(mountain1,None)
cv2.drawKeypoints(m1_clr,keypoints_mountain1,m1_clr)
cv2.imwrite('output/task1/mountain_1_sift.jpg',m1_clr)

keypoints_mountain2,m2_des = sift.detectAndCompute(mountain2,None)
cv2.drawKeypoints(m2_clr,keypoints_mountain2,m2_clr)
cv2.imwrite('output/task1/mountain_2_sift.jpg',m2_clr)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(m1_des,m2_des, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(m1_clr,keypoints_mountain1,m1_clr,keypoints_mountain1,good,None,flags=2)
cv2.imwrite("output/task1/sift_knn.jpg",img3)

