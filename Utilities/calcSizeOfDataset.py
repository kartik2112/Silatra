from os import listdir

for digit1 in range(0,10):
    dirPath = "../training-images/Digits/"+str(digit1)+"/Right_Hand/Normal/"
    print("Digits/"+str(digit1)+" - "+str(len(listdir(dirPath)))+" no of images")