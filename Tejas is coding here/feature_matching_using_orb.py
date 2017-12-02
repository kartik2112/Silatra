import cv2
import numpy as np

file_path = '../training-images/Digits/1/Right_Hand/Normal/'
print('\nEnter image numbers for Right hand, normal and sign as 1\nEg. 2,3 (No space in between)\n')
image_number_1 = input('Image inputs: ')
image_number_1, image_number_2 = image_number_1.split(',')

image1 = cv2.imread(file_path+image_number_1+'.png')                                  # Query
image2 = cv2.imread(file_path+image_number_2+'.png')                                  # Train

# Convert images to grayscale
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (For contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
image1 = clahe.apply(image1)
image2 = clahe.apply(image2)


""" # FLANN Based feature matching

# For FLANN based feature matching, we can use only SIFT.

sift = cv2.xfeatures2d.SIFT_create()
key_points_1, descriptors_1 = sift.detectAndCompute(image1, None)
key_points_2, descriptors_2 = sift.detectAndCompute(image2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

# Need to draw only good matches, so create a mask
matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

final_image = cv2.drawMatchesKnn(image1,key_points_1,image2,key_points_2,matches,None,**draw_params)

"""

# Brute force matching using ORB descriptors

# Detect keypoints and draw descriptors
orb = cv2.ORB_create()
key_points_1, descriptors_1 = orb.detectAndCompute(image1, None)
key_points_2, descriptors_2 = orb.detectAndCompute(image2, None)

# create BFMatcher object
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf_matcher.match(descriptors_1,descriptors_2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
final_image = cv2.drawMatches(image1,key_points_1,image2,key_points_2,matches[:10], None, flags=2)

cv2.imshow('Matches', final_image)
cv2.waitKey(1000000)
cv2.destroyAllWindows()