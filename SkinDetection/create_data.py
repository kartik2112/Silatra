import cv2 as cv2

def extract_hsv_features(file, label):
    data = set()
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for row in img:
        for pixel in row:
            s=''
            for hsv_values in pixel:
                s += str(hsv_values) + '\t'
            s += label + '\n'
            data.add(s)
    return data

if __name__=="__main__":
    '''
    mark_data_as = 1 for skin sample
    mark_data_as = 2 for non-skin sample
    '''
    print('Extracting features... Completed: 0%\r',end='')
    hsv_data = set()
    img_template = 'Test_Images/Samples for training/'
    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 8, '.jpg', '1'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)

    sample_start_number, sample_end_number, img_ext, mark_data_as = 6, 10, '.png', '1'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)
    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 9, '.jpg', '2'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Non-skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Non-skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)

    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 3, '.png', '2'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Non-skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Non-skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)

        
    print('                                                                                      \r',end='')
    print('Extraction complete! Saving...\r',end='')
    with open('hsv.data','w') as data_file:
        for row in hsv_data: data_file.write(row)
    print('Data is ready for building the model!')