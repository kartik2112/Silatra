from os import listdir, system
from os.path import isfile

digits=[0,1,2,3,4,5,6,7,8,9]

for digit in digits:
    dest_path="C:\\Users\\VR-Admin\\Pictures\\training-images\\Digits_Kartik\\"+str(digit)+"\\Right_Hand\\Normal\\"
    orig_path="C:\\Users\\VR-Admin\\Pictures\\training-images\\Digits_Varun\\"+str(digit)+"\\Right_Hand\\Normal\\"
    files = listdir(orig_path)
    files.sort()
    newCount = len(listdir(dest_path)) + 1
    print(str(newCount-1)+" images present in "+dest_path)
    for im_file in files:
        comm_str="copy "+orig_path+im_file+" "+dest_path+str(newCount)+".png"
        # print(comm_str)
        system(comm_str)
        # break
        newCount+=1
    # break
    print("\nCopied files from "+orig_path+" to "+dest_path)
