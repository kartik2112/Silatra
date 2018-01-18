#%%
import cv2, time, argparse, matplotlib.pyplot as plt
from numpy import array,uint8,hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

''' ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help='Use this flag followed by image file to do segmentation on an image')
args = vars(ap.parse_args()) '''

n, true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0, 0
test_skin_samples, test_non_skin_samples = 0, 0

MODULUS = 1
data, attributes_info, classes, desired_values = [], [{}, {}, {}], [0,0], []
train_data, test_data, train_labels, test_labels=[],[],[],[]
ranges=[179.0,255.0,255.0]
with open('skin-detection-training.txt') as f:
	row_count=1
	print('Reading the Silatra training dataset...\r',end='')
	while True:
		# Read attributes_info row by row
		line = f.readline()
		if line == '': break                    # End of file
		line = line.split(',')
		pixel = line[0:len(line)-1]             # attributes_info comes as: [h,s,v,class].
		for i in range(len(pixel)): pixel[i] = float(pixel[i])*1.0/ranges[i]
		train_data.append(pixel)
		desired_value = int(line[len(line)-1]) #- 1
		train_labels.append(desired_value)

with open('skin-detection-testing.txt') as f:
	row_count=1
	print('Reading the Silatra testing dataset...\r',end='')
	while True:
		# Read attributes_info row by row
		line = f.readline()
		if line == '': break                    # End of file
		line = line.split(',')
		pixel = line[0:len(line)-1]             # attributes_info comes as: [h,s,v,class].
		for i in range(len(pixel)): pixel[i] = float(pixel[i])*1.0/ranges[i]
		test_data.append(pixel)
		desired_value = int(line[len(line)-1]) #- 1
		test_labels.append(desired_value)

#%%
# train_data, test_data, train_labels, test_labels = train_test_split(data, desired_values, test_size=0.3, random_state=31)
print('                                                             \r',end='')
print('Just a minute...\r',end='')
for i in range(len(train_data)):
	pixel = train_data[i]
	desired_value = train_labels[i]

	# Build count for each bin and desired value
	classes[desired_value] += 1
	n += 1
	for i in range(len(pixel)):
		near_val = pixel[i]
		try: attributes_info[i][near_val][desired_value] += 1
		except: attributes_info[i][near_val] = [0,0]

for attribute in attributes_info:
	for key in attribute.keys():
		attribute[key][0] = float(attribute[key][0])/classes[0]
		attribute[key][1] = float(attribute[key][1])/classes[1]
classes[0] = float(classes[0]) / n
classes[1] = float(classes[1]) / n
print('Model is ready\r',end='')

#%%
print('Testing in process...\r',end='')
model_predictions=[]
for i in range(len(test_data)):
	pixel = test_data[i]
	desired_value = test_labels[i]

	probability_skin, probability_non_skin, prediction = classes[0], classes[1], 0
	for channel in range(len(pixel)):
		value = pixel[channel]
		probability_skin *= attributes_info[channel][value][0]
		probability_non_skin *= attributes_info[channel][value][1]

	if probability_skin <= probability_non_skin: prediction = 1

	if desired_value is 0: test_skin_samples += 1
	else: test_non_skin_samples += 1

	if desired_value is 0 and prediction is 0: true_positives += 1
	elif desired_value is 0 and prediction is 1: false_positives += 1
	elif desired_value is 1 and prediction is 0: false_negatives += 1
	elif desired_value is 1 and prediction is 1: true_negatives += 1
	model_predictions.append(prediction)
print('Testing complete\r',end='')

#%%
print('Model characterisitics: \n')
print('Confusion matrix:    \n\n---------------------------------\n|  C  |\tS\t|\tNS\t|\n---------------------------------')
print('|  S  |\t'+str(true_positives)+'\t|\t'+str(false_positives)+'\t|')
print('|  NS |\t'+str(false_negatives)+'\t|\t'+str(true_negatives)+'\t|',end='\n')
print('---------------------------------')
correct_predictions = true_positives + true_negatives
print('Accuracy: '+str(correct_predictions)+' / '+str(len(test_data))+' = '+str(round(float(correct_predictions/len(test_data)), 4)*100)+'%')

sensitivity = true_positives*1.0/test_skin_samples
specificity = true_negatives*1.0/test_non_skin_samples
precision = true_positives*1.0/(true_positives+false_negatives)
f1_score = 2*true_positives*1.0/(2*true_positives+false_positives+false_negatives)

print('Sensitivity of model: %0.4f' % sensitivity)
print('Specificity of model: %0.4f' % specificity)
print('Precision of model: %0.4f' % precision)
print('F1 score of model: %0.4f' % f1_score)

#%%
fpr,tpr,_ = roc_curve(test_labels, model_predictions)
roc_auc = auc(fpr, tpr)
plt.title('ROC for Naive Bayesian')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('\nProbabilistic model ready!')

#%%
image_file = 'Test_Images/'
image_file += input('\nInput image: ')
''' if not args.get('image'): image_file += input('\nInput image: ')
else:
	image_file += args.get('image')
	print('\nUsing: '+image_file) '''

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
for i in range(len(image)):                                   # Each row
	for j in range(len(image[i])):                            # Each pixel
		for k in range(3):                                  # Each channel (h/s/v)
			image[i][j][k] = image[i][j][k]*1.0/ranges[k]

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
			value = pixel[channel]
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


''' def predict(row):
	prediction = []
	for pixel in row:
		probability_skin, probability_non_skin = classes[0] ,classes[1]
		for channel in range(3):
			value = pixel[channel] - pixel[channel]%MODULUS
			probability_skin *= attributes_info[channel][value][0]
			probability_non_skin *= attributes_info[channel][value][1]
		prediction.append([probability_skin, probability_non_skin])
	return prediction

upper_row_predictions, curr_row_predictions, lower_row_predictions = [], predict(image[0]), predict(image[1])
n, k1, K = len(image[0]), 1.0, 1
for i in range(len(image)):
	for j in range(len(curr_row_predictions)):
		l_skin, count = 0.0, 0
		if i is not 0:
			count += 1
			if j > 0:
				l_skin = upper_row_predictions[j-1][0]
				count += 1
			l_skin += upper_row_predictions[j][0]
			if j < n-1:
				l_skin += upper_row_predictions[j+1][0]
				count += 1
		if j > 0:
			l_skin += curr_row_predictions[j-1][0]
			count += 1
		if j < n-1:
			l_skin += curr_row_predictions[j+1][0]
			count += 1
		if i is not len(image)-1:
			count += 1
			if j > 0:
				l_skin += lower_row_predictions[j-1][0]
				count += 1
			l_skin += lower_row_predictions[j][0]
			if j < n-1:
				l_skin += lower_row_predictions[j+1][0]
				count += 1
		alpha = l_skin
		l_skin = l_skin*k1/(1.0*count)
		if curr_row_predictions[j][0]*l_skin >= 0.5: k1 = count*1.0*K/alpha
		else:
			image[i][j] = [0.0,0.0,0.0]
			k1 = 1
		pixels_processed += 1
		if pixels_processed%10000 == 0: print('Pixels processed: '+str(pixels_processed/1000)+'k / '+str(total_pixels/1000)+'k\r',end='')
	upper_row_predictions = curr_row_predictions
	curr_row_predictions = lower_row_predictions
	if i < len(image)-2: lower_row_predictions = predict(image[i+2]) '''

time_for_image = time.time() - time_for_image
#print('\n\nTime required per pixel = '+str(time_per_pixel)+' seconds')
print('Time required for segmentation = '+str(time_for_image)+' seconds')
cv2.imshow('Segmentation results',hstack([original, cv2.cvtColor(array(image, uint8), cv2.COLOR_HSV2BGR)]))
print()
cv2.waitKey(100000)
cv2.destroyAllWindows()
