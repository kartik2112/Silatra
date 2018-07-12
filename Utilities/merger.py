# This is used for copying images from one folder to another.
# This is done by counting no. of images in destination folder,
#   then renaming the source images starting from this count.
# Thus, the source images don't replace destination images because of renaming.

from os import listdir, system
from os.path import isfile

digit1 = 5

# Path from where to copy images (Source)
# newDataPath = "../training-images/Digits_Kartik/"+str(digit1)+"/Right_Hand/Normal/"
newDataPath = "/media/kartik/Kartik SK 1TB/Silatra/tejas/i/"

files = listdir(newDataPath)
files.sort()
# print(files)


# Path where you want to add more images (Destination)
# mergeDataPath = "../training-images/Digits/"+str(digit1)+"/Right_Hand/Normal/"
mergeDataPath = "/media/kartik/Kartik SK 1TB/Silatra/Letters/i/"

newCount = len(listdir(mergeDataPath)) + 1

print((newCount-1)," images present in "+mergeDataPath)

print("To start transfer, you need to manually edit the program")




for file1 in files:
    system("cp '"+newDataPath+file1+"' '"+mergeDataPath+str(newCount)+".png'")
    newCount+=1
print("\nCopied files from "+newDataPath+" to "+mergeDataPath)
