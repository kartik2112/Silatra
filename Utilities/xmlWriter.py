# Reference: https://handmap.github.io/dlib-classifier-for-object-detection/

import os
import cv2

folders = open('dlibImageSources.txt')
f = open('hand-dataset.xml','a')

# folders = []
# for i in range(0,10):
#     folders += [str(i)]
# for i in range(0,27):
#     folders += [str(chr(i+ord('a')))]    

imgNo = 1
for folderPath in folders:
    
    # ctr = 120
    # print(os.listdir("images/"+folderPath))
    # imgsPaths = os.listdir("images/"+folderPath)
    imgsPaths = os.listdir(folderPath)
    imgsPaths.sort()
    for img1 in imgsPaths:
        # filename = "images/"+folderPath+"/"+img1
        filename = folderPath+"/"+img1
        img123 = cv2.imread(filename)
        try:
            HEIGHT, WIDTH, _ = img123.shape
        except:
            print("SKIPPING")
            print(folderPath+"/"+img1)
            continue
        print(folderPath+"/"+img1)
        f.write("<image file='%s'>\n\t<box top='%d' left='%d' width='%d' height='%d'>\n\t\t<label>%s</label>\n\t</box>\n</image>\n"%(img1,0,0,WIDTH,HEIGHT,"hand"))