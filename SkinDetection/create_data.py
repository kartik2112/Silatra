import cv2 as cv2
from numpy import uint8

def extract_hsv_features(file, label):
    data = []
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for row in img:
        for pixel in row:
            s=''
            for hsv_values in pixel:
                s += str(hsv_values) + '\t'
            s += label + '\n'
            data.append(s)
    return data

def convert_uci_data(file):
    data = []
    with open(file) as f:
        while True:
            line = f.readline()    
            if line is '': break
            line = line.split('\t')
            pixel = line[0:len(line)-1]
            label = line[len(line)-1]
            pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
            s=''
            for hsv_values in pixel:
                s += str(hsv_values) + '\t'
            s += label
            data.append(s)
        return data

if __name__=="__main__":
    '''
    mark_data_as = 1 for skin sample
    mark_data_as = 2 for non-skin sample

    Original end points - 8, 5, 9, 3
    '''
    hsv_data = set()
    img_template = 'Test_Images/Samples for training/'
    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 13, '.jpg', '1'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)

    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 6, '.png', '1'
    for i in range(sample_start_number,sample_end_number+1):
        print('                                                                                      \r',end='')
        print('Processing: '+img_template+'Skin samples/'+str(i)+img_ext+'\r',end='')
        extracted_data = extract_hsv_features(img_template+'Skin samples/'+str(i)+img_ext,mark_data_as)
        for row in extracted_data: hsv_data.add(row)

    sample_start_number, sample_end_number, img_ext, mark_data_as = 1, 15, '.jpg', '2'
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
    #hsv_data = convert_uci_data('uci_skin_segmentation_data.txt')
    print('Extraction complete! Saving...\r',end='')
    with open('data.txt','w') as data_file:
        #for row in uci_data: data_file.write(row)
        for row in hsv_data: data_file.write(row)
    print('Data is ready for building the model!')