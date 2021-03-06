import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.stats import zscore

index = []

def csv_read(file_name):
	import csv
	emgdata = []
	with open(file_name, 'rt') as f :
		data = csv.reader(f)
		for row in data:
			emgdata.append(row)
	emgdata = np.array(emgdata).astype(int)
	return emgdata

def read_data(zero_padding=True,normalize=True):
	main_dir = os.getcwd()
	filename = "*/*/*.csv" 
	files = []
	labels = []
	words = []
	length = []
	SR =600
	"""
	reading all the audio filenames into a list
	#"""
	ind = 'start'
	countt= 0
	for file in glob.glob(filename):
		files.append(file)
		# print(file)
		labels.append(file.split("/")[1])
		if ind != file.split("/")[1] :
			index.append(countt)
			ind = file.split("/")[1]

		emg = csv_read(file)
		words.append(emg)
		length.append(emg.shape[1])
		countt = countt + 1

	len_max_array = max(length)
	len_min_array = min(length)
	
	"""
	zero padding all the audio files so that each file equal number of members in the array
	"""
	if(zero_padding):
		for i in range(len(words)):
			diff = len_max_array - (words[i].shape[1])
			if(diff>0):
				# np.pad(matrix, ((<row above>, <row below>), (<column front>, <column back>), <pad with ie minimum, zero, maximum, mean, etc >)
				words[i] = np.pad(words[i], ((0, 0), (int(diff/2),int(diff-diff/2))),'constant', constant_values = 0)

	if(normalize):
		# words = [zscore(word) for word in words]
		words = [zscore(word,axis = 1) for word in words]

	return words,(labels),SR,len_max_array

words, labels, SR, len_max_array = read_data()

print('no of files : ',len(words))
# print('shape of the data : ',words.shape)
print('no of samples : ',words[0].shape)
print(' sample : ',words[0])

def collect_equal_data(words, labels):
	tmp_labels = list(labels)
	counter = []
	for lab in np.unique(tmp_labels):
		counter.append(tmp_labels.count(lab))
	print(counter)
	min_data_number = np.min(counter)
	equal_words = []
	equal_labels = []
	for indy in index:
		equal_words.extend(words[indy: indy+ min_data_number])
		equal_labels.extend(labels[indy: indy+min_data_number])
	print('finished equalization')
	return equal_words, equal_labels

from features import *
def call_feature(words,window_size , sampling_rate ,len_max_array ):
	window_size = 30 #ms
	f = Features(window_size = window_size, sampling_rate = sampling_rate,len_max_array = len_max_array)
	dummy_words = []
	# for i in range(len(words)):
	for i in range(len(words)):
		dummy_words.append(f.compute_temporal_features(word = words[i]))
		print('checking word', i)
	return np.array(dummy_words)

	# f = Features()

window_size = 30
collect_feature = not True
save_or_load_feature = not True

if collect_feature:
	words, labels = collect_equal_data(words, labels)
	words = call_feature(words, window_size = window_size,sampling_rate   = SR, len_max_array = len_max_array)

import pickle
if save_or_load_feature:
	outfile = open('feat_words.pickle','w')
	pickle.dump(words, outfile)
	outfile.close()
	outfile = open('feat_labels.pickle','w')
	pickle.dump(labels, outfile)
	outfile.close()
else :
	infile = open('feat_words.pickle','r')
	words = pickle.load(infile)
	infile.close()
	infile = open('feat_labels.pickle','r')
	labels = pickle.load(infile)
	infile.close()

print('no of files : ', type(words))
# print('shape of the data : ',words.shape)
print('no of samples : ',words[0].shape)
print(' sample : ',words[0])


'''
from features import *
f = Features(words, window_size, SR, len_max_array)
feature = (f.compute_temporal_features())

print(np.array(feature).shape)

tmp_labels = list(labels)
counter = []
for lab in np.unique(tmp_labels):
	counter.append(tmp_labels.count(lab))
print(counter)
mindatanumber = np.min(counter)
new_feat= []
new_label = []
left_feat = []
left_label = []
# for indy in index:
# 	new_feat.extend(feature[indy:indy+mindatanumber])
# 	new_label.extend(labels[indy:indy+mindatanumber])


# 0------10------15-------20------25

for i in range(len(index)):
	new_feat.extend(feature[index[i] : index[i]+mindatanumber ])
	if i >= len(index)-1:
		left_feat.extend(feature[index[i]+mindatanumber : -1])
	else:
		left_feat.extend(feature[index[i]+mindatanumber : index[i+1]])
	new_label.extend(labels[index[i] : index[i]+mindatanumber ])
	if i >= len(index)-1:
		left_label.extend(labels[index[i]+mindatanumber : -1])
	else:
		left_label.extend(labels[index[i]+mindatanumber : index[i+1]])

feature = new_feat
labels = new_label

print(np.array(feature).shape)


import random 
mapindexpos = list(zip(feature, labels))
random.shuffle(mapindexpos)
feature, labels = zip(*mapindexpos)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(labels)
left_label = labelencoder_y.fit_transform(left_label)

print(labels)
print(left_label)

feature = np.array(feature)
labels = np.array(labels)

train_data = feature
train_label = labels
test_data = np.array(left_feat)
test_label = np.array(left_label)

print(feature.shape)

feature = np.reshape(feature, (feature.shape[0], feature.shape[1], 1))
train_data = np.reshape(train_data,(train_data.shape[0], train_data.shape[1], 1))
test_data = np.reshape(test_data,(test_data.shape[0], test_data.shape[1],1))
print(feature.shape)

# from sklearn.model_selection import train_test_split
# train_data, test_data, train_label, test_label = train_test_split(feature, labels, test_size = 0.10)

from keras.models import Sequential 
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
import keras.regularizers

clf = Sequential()
clf.add(Convolution1D(32,5, input_shape = (feature.shape[1], 1), activation = 'relu'))
clf.add(MaxPooling1D(pool_size = 5))
clf.add(Flatten())
clf.add(Dense(output_dim = 128, activation = 'relu')) #, kernel_regularizer = keras.regularizers.l1(1e-5)))
clf.add(Dense(output_dim = 10, activation = 'sigmoid'))
clf.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
clf.summary()

history = clf.fit(train_data, train_label, validation_split = 0.2, epochs = 20, batch_size = 32)
test_loss, test_accuracy = clf.evaluate(test_data,test_label)
print("Tested accuracy: ", test_accuracy)
print("Tested loss: ", test_loss)
pred = clf.predict(test_data)

print(history.history.keys())
# print(pred.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test_validation'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test_validation'], loc = 'upper left')
plt.show()
plt.close()

pred_pred = []
points = 0 
for i in range(len(pred)):
	print(int(np.argmax(pred[i])),int(test_label[i]))
	pred_pred.append(np.argmax(pred[i]))
	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
		points+=1
print("Accuracy : ", ( (points/ len(pred)) * 100))

pred_pred = np.array(pred_pred)

from sklearn.metrics import confusion_matrix
unique_labels = np.unique(labels)
cm = confusion_matrix(test_label, pred_pred, [1,2,3,4,5,6,7,8,9,10])
print(cm)
fig= plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + list(unique_labels))
ax.set_yticklabels([''] + list(unique_labels))
plt.xlabel('predicted')
plt.ylabel('True')
plt.show()
plt.close()

# plt.pcolormesh(time, frequency, spectrogram)
# plt.ylabel('frequency')
# plt.xlabel('time')
# plt.show()
#https://stackoverflow.com/questions/45100189/specify-number-of-samples-in-a-segment-for-short-time-fourier-transform-using-sc
#https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
#'''