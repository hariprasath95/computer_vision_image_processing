UBIT = 'hparthas'

import cv2
from matplotlib import pyplot as plt
import numpy as np
import random 
np.random.seed(sum(ord(c) for c in UBIT)) 
# reference https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
def warpTwoImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() + 20)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 20)

    Ht = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
    Ht1 = Ht.dot(H)
    Ht1[0][2] = Ht1[0][2]
    Ht1[1][2] = Ht1[1][2]
    result = cv2.warpPerspective(img2, Ht1, (xmax-xmin, ymax-ymin))
    result[-ymin:h1+(-ymin),-xmin:w1+(-xmin)] = img1
    return result
# read image 1 and convert to BW
m1_clr = cv2.imread('data/mountain1.jpg')
mountain1= cv2.cvtColor(m1_clr,cv2.COLOR_BGR2GRAY)

# read image 2 and convert to BW
m2_clr = cv2.imread('data/mountain2.jpg')
mountain2 = cv2.cvtColor(m2_clr,cv2.COLOR_BGR2GRAY)

# Extract Sift features and compute Descriptors for image 1 and image 2
sift = cv2.xfeatures2d.SIFT_create()
keypoints_mountain1 ,m1_des= sift.detectAndCompute(mountain1,None)
image1_withkp = cv2.drawKeypoints(m1_clr,keypoints_mountain1,None)
cv2.imwrite('output/task1/task1_sift1.jpg',image1_withkp)

keypoints_mountain2,m2_des = sift.detectAndCompute(mountain2,None)
image2_withkp =  cv2.drawKeypoints(m2_clr,keypoints_mountain2,None)
cv2.imwrite('output/task1/task1_sift2.jpg',image2_withkp)

bf = cv2.BFMatcher()
matches = bf.knnMatch(m1_des,m2_des, k=2)

good_knnmatch = []
good_filter = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_knnmatch.append(([m]))
        good_filter.append((m))

img3 = cv2.drawMatchesKnn(m1_clr,keypoints_mountain1,m2_clr,keypoints_mountain2,good_knnmatch,None,flags=2)
cv2.imwrite("output/task1/task1_matches_knn.jpg",img3)

src_pts = np.float32([ keypoints_mountain1[m[0].queryIdx].pt for m in good_knnmatch ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints_mountain2[m[0].trainIdx].pt for m in good_knnmatch ]).reshape(-1,1,2)

M, mask = cv2.findHomography(dst_pts,src_pts , cv2.RANSAC,3.0)
print(M)

matchesMask = mask.ravel().tolist()
h,w,d = m1_clr.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

selected_matches = [0] * len(matchesMask)
matches_count = 0 
while  matches_count < 10 :
    select_index = np.random.randint(0,len(matchesMask))
    if selected_matches[select_index] == 0 and matchesMask[select_index] == 1:
        selected_matches[select_index] = 1
        matches_count =matches_count+1

draw_params = dict(matchColor = (255,0,0), 
                   singlePointColor = None,
                   matchesMask = selected_matches, 
                   flags = 2)

img3 = cv2.drawMatches(m1_clr,keypoints_mountain1,m2_clr,keypoints_mountain2,good_filter,None,**draw_params)

cv2.imwrite("output/task1/task1_matches.jpg",img3)
im_out =warpTwoImages(m2_clr,m1_clr,np.linalg.inv(M))

cv2.imwrite("output/task1/task1_pano.jpg",im_out)