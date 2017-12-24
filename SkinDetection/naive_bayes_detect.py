import cv2, time, argparse
from numpy import array,uint8,hstack
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help='Use this flag followed by image file to do segmentation on an image')
args = vars(ap.parse_args())

n=0
def read_uci_data():
	print('Reading the UCI dataset...\r',end='')
	with open('uci_skin_segmentation_data.txt') as f:
		while True:
			# Read attributes_info row by row
			line = f.readline()
			if line == '': break                    # End of file
			line = line.split('\t')
			pixel = line[0:len(line)-1]             # attributes_info comes as: [h,s,v,class].
			for i in range(len(pixel)): pixel[i] = int(pixel[i])

			# Conversion of BGR to HSV colour space.
			pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
			data.append(pixel)
			desired_value = int(line[len(line)-1]) - 1
			desired_values.append(desired_value)

def read_silatra_data():
	with open('silatra_dataset_complete.txt') as f:
		row_count=1
		print('Reading the Silatra dataset...\r',end='')
		while True:
			# Read attributes_info row by row
			line = f.readline()
			if line == '': break                    # End of file
			line = line.split('\t')
			pixel = line[0:len(line)-1]             # attributes_info comes as: [h,s,v,class].
			for i in range(len(pixel)): pixel[i] = int(pixel[i])
			data.append(pixel)
			desired_value = int(line[len(line)-1]) - 1
			desired_values.append(desired_value)

def build_data():
	global n
	for i in range(len(train_data)):
		pixel = train_data[i]
		desired_value = train_labels[i]

		# Build count for each bin and desired value
		classes[desired_value] += 1
		n += 1
		for i in range(len(pixel)):
			near_val = pixel[i] - pixel[i]%MODULUS
			try: attributes_info[i][near_val][desired_value] += 1
			except: attributes_info[i][near_val] = [0,0]

def build_probabilities():
	global n
	for attribute in attributes_info:
		for key in attribute.keys():
			attribute[key][0] = float(attribute[key][0])/classes[0]
			attribute[key][1] = float(attribute[key][1])/classes[1]
	classes[0] = float(classes[0]) / n
	classes[1] = float(classes[1]) / n

def evaluate():
	true_positives, true_negatives, false_positives, false_negatives = 0,0,0,0
	for i in range(len(test_data)):
		pixel = test_data[i]
		desired_value = test_labels[i]

		probability_skin, probability_non_skin, prediction = classes[0], classes[1], 0
		for channel in range(len(pixel)):
			value = pixel[channel] - pixel[channel]%MODULUS
			probability_skin *= attributes_info[channel][value][0]
			probability_non_skin *= attributes_info[channel][value][1]
		
		if probability_skin <= probability_non_skin: prediction = 1
		if desired_value is 0 and prediction is 0: true_positives += 1
		elif desired_value is 0 and prediction is 1: true_negatives += 1
		elif desired_value is 1 and prediction is 0: false_positives += 1
		elif desired_value is 1 and prediction is 1: false_negatives += 1
	
	print('Confusion matrix:    \n\n---------------------------------\n|  C  |\tS\t|\tNS\t|\n---------------------------------')
	print('|  S  |\t'+str(true_positives)+'\t|\t'+str(true_negatives)+'\t|')
	print('|  NS |\t'+str(false_positives)+'\t|\t'+str(false_negatives)+'\t|',end='\n')
	print('---------------------------------')
	correct_predictions = true_positives + false_negatives
	print('Accuracy: '+str(correct_predictions)+' / '+str(len(test_data))+' = '+str(round(float(correct_predictions/len(test_data)), 4)*100)+'%')

if __name__=="__main__":
	MODULUS = 1
	data, attributes_info, classes, desired_values = [], [{}, {}, {}], [0,0], []
	#read_uci_data()
	read_silatra_data()

	train_data, test_data, train_labels, test_labels = train_test_split(data, desired_values, test_size=0.3, random_state=31)
	print('                                                             \r',end='')
	print('Just a minute....\r',end='')
	build_data()
	build_probabilities()

	print('Testing in process...\r',end='')
	evaluate()

	print('\nProbabilistic model ready!')
	image_file = 'Test_Images/'
	if not args.get('image'): image_file += input('\nInput image: ')
	else:
		image_file += args.get('image')
		print('\nUsing: '+image_file)

	image = cv2.imread(image_file)
	if float(len(image)/len(image[0])) == float(16/9): image = cv2.resize(image, (180,320))
	elif float(len(image)/len(image[0])) == float(9/16): image = cv2.resize(image, (320,180))
	elif float(len(image)/len(image[0])) == float(4/3): image = cv2.resize(image, (240,320))
	elif float(len(image)/len(image[0])) == float(3/4): image = cv2.resize(image, (320,240))
	elif float(len(image)/len(image[0])) == 1: image = cv2.resize(image, (250,250))
	else: image = cv2.resize(image, (250,250))

	total_pixels = len(image)*len(image[0])
	print('\n'+str(len(image))+'x'+str(len(image[0]))+'='+str(total_pixels)+' pixels')
	original = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image = image.tolist()

	print('Segmentation is starting...\r',end='')
	binary_image = []
	pixels_processed = 0
	time_for_image = time.time()
	time_per_pixel = 0

	for row in image:
		binary_row = []
		for pixel in row:
			probability_skin, probability_non_skin = classes[0] ,classes[1]
			pixels_processed += 1
			time_per_pixel = time.time()
			for channel in range(3):
				value = pixel[channel] - pixel[channel]%MODULUS
				probability_skin *= attributes_info[channel][value][0]
				probability_non_skin *= attributes_info[channel][value][1]
			time_per_pixel = time.time() - time_per_pixel
			if probability_skin > probability_non_skin: binary_row.append([0,0,255])
			else: binary_row.append([0,0,0])

			if pixels_processed%10000 == 0: print('Pixels processed: '+str(pixels_processed/1000)+'k / '+str(total_pixels/1000)+'k\r',end='')
		binary_image.append(binary_row)

	print('Pixels processed: '+str(total_pixels/1000)+'k / '+str(total_pixels/1000)+'k\r',end='')
	for i in range(len(image)):
		for j in range(len(image[i])):
			for k in range(3): image[i][j][k] = float(binary_image[i][j][k])

	time_for_image = time.time() - time_for_image
	print('\n\nTime required per pixel = '+str(time_per_pixel)+' seconds')
	print('Time required for segmentation = '+str(time_for_image)+' seconds')
	cv2.imshow('Segmentation results',hstack([original, cv2.cvtColor(array(image, uint8), cv2.COLOR_HSV2BGR)]))
	print()
	cv2.waitKey(100000)
	cv2.destroyAllWindows()