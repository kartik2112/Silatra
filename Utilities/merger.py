from os import listdir, system
from os.path import isfile

digit1 = 5

# newDataPath = "../training-images/Digits_Kartik/"+str(digit1)+"/Right_Hand/Normal/"
newDataPath = "../training-images/Digits/0/"
Fist
files = listdir(newDataPath)
files.sort()
# print(files)

# mergeDataPath = "../training-images/Digits/"+str(digit1)+"/Right_Hand/Normal/"
mergeDataPath = "../training-images/Gesture_Signs/That_Is_Good_Circle/"

newCount = len(listdir(mergeDataPath)) + 1

print((newCount-1)," images present in "+mergeDataPath)

print("To start transfer, you need to manually edit the program")




for file1 in files:
    system("cp "+newDataPath+file1+" "+mergeDataPath+str(newCount)+".png")
    newCount+=1
print("\nCopied files from "+newDataPath+" to "+mergeDataPath)