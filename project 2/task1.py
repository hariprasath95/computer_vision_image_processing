import cv2
from matplotlib import pyplot as plt
import numpy as np
def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel())

    Ht = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]]) # translate
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

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(m1_des,m2_des, k=2)

# Apply ratio test
good_knnmatch = []
good_filter = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_knnmatch.append(([m]))
        good_filter.append((m))

img3 = cv2.drawMatchesKnn(m1_clr,keypoints_mountain1,m2_clr,keypoints_mountain2,good_knnmatch,None,flags=2)
cv2.imwrite("output/task1/sift_knn.jpg",img3)

src_pts = np.float32([ keypoints_mountain1[m[0].queryIdx].pt for m in good_knnmatch ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints_mountain2[m[0].trainIdx].pt for m in good_knnmatch ]).reshape(-1,1,2)

M, mask = cv2.findHomography(dst_pts,src_pts , cv2.RANSAC,3.0)

matchesMask = mask.ravel().tolist()
h,w,d = m1_clr.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(mountain1,keypoints_mountain1,mountain2,keypoints_mountain2,good_filter,None,**draw_params)
im_out =warpTwoImages(m2_clr,m1_clr,np.linalg.inv(M))

cv2.imwrite("output/task1/task1_pano.jpg",im_out)