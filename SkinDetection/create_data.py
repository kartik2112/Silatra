import cv2 as cv2

def extract_hsv_features(file, label):
    data = ''
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for row in img:
        for pixel in row:
            for hsv_values in pixel:
                data += str(hsv_values) + '\t'
            data += label + '\n'
    return data

if __name__=="__main__":
    print('Extracting features... Completed: 0%\r',end='')
    hs_data = ''
    img_template, img_ext, intended_label = 'Test_Images/Samples for training/sample', '.png', '1'
    total_samples, curr_sample = 7, 1
    for i in range(total_samples):
        hs_data += extract_hsv_features(img_template+str(curr_sample)+img_ext,intended_label)
        print('Extracting features... Completed:'+str(round(float(curr_sample)/total_samples,3))+'%\r',end='')
    print('Extrated features... Completed: 100%\tSaving..')
    with open('hs_new.data','w') as data_file: data_file.write(hs_data)