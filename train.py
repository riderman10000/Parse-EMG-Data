import pandas as pd 
import sklearn
import numpy as np 
from features import *
import os
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.model_selection import train_test_split



index = []

def read_data(zero_padding=True,normalize=True):
	main_dir = os.getcwd()
	filename = "*/*/*.wav" 
	files = []
	labels = []
	words = []
	length = []
	SR = 0
	"""
	reading all the audio filenames into a list
	"""
	ind = 'start'
	countt= 0
	for file in glob.glob(filename):
		files.append(file)
		# print(file)
		labels.append(file.split("/")[1])
		if ind != file.split("/")[1] :
			index.append(countt)
			ind = file.split("/")[1]
		# print("with label", labels)
		SR, audio = wavfile.read(file)
		words.append(audio)
		length.append(len(audio))
		countt = countt + 1

	len_max_array = max(length)
	len_min_array = min(length)
	
	"""
	zero padding all the audio files so that each file equal number of members in the array
	"""
	if(zero_padding):
		for i in range(len(words)):
			diff = len_max_array - len(words[i])
			if(diff>0):
				words[i] = np.pad( words[i] ,(int(diff/2),int(diff-diff/2)))
	if(normalize):
		words = [zscore(word) for word in words]

	return words,(labels),SR,len_max_array

class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]
window_size = 10 #in milliseconds
# seg_size = 160 #equivalent sample size of the window_size

words,labels,SR,len_max_array = read_data()




print(labels)

labels = np.array(labels)

f = Features(words,window_size,SR,len_max_array)
feat = (f.compute_temporal_features())
# feat = f.compute_spectral_features()

print(feat[0])
print(feat.shape)
tmp_labels = list(labels)
counter = []
for lab in np.unique(tmp_labels):
	counter.append(tmp_labels.count(lab))
print(counter)
mindatanumber = np.min(counter)
new_feat= []
new_label = []
for indy in index:
	new_feat.extend(feat[indy:indy+mindatanumber])
	new_label.extend(labels[indy:indy+mindatanumber])

feat = new_feat
labels = new_label


print(len(feat))
print(labels)
tmp_labels = list(labels)
counter = []
for lab in np.unique(tmp_labels):
	counter.append(tmp_labels.count(lab))
print(counter)

import random
mapindexpos = list(zip(feat, labels))
random.shuffle(mapindexpos)
feat, labels = zip(*mapindexpos)

feat = np.array(feat)
labels = np.array(labels)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(labels)

#normalize

train_data, test_data, train_label, test_label = train_test_split(feat,labels,test_size=0.20)
# print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

print(train_label[100])
print(test_label[50])

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators = 1000, max_depth = 200)

# # from sklearn.svm import SVC 
# # clf = SVC(kernel = 'rbf', random_state = 0) 
# history = clf.fit(train_data,train_label)
# # print(history.history.keys())

# pred = clf.predict(test_data)

# points = 0
# for i in range(len(test_label)):
# 	if pred[i] == test_label[i]:
# 		points += 1

# print("Accuracy : ", ( (points/ len(pred)) * 100))


import tensorflow as tf 
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation = "relu" , input_shape=(train_data.shape[1],)))
# model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Dense(32, activation = "relu"))
# model.add(keras.layers.Dropout(0.10))
model.add(keras.layers.Dense(10 , activation = "softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(train_label)

print("label", train_label)


history = model.fit(train_data,train_label,validation_split = 0.2,epochs=20,batch_size=32)

test_loss, test_accuracy = model.evaluate(test_data,test_label)
print("Tested accuracy: ", test_accuracy)
print("Tested loss: ", test_loss)

pred = model.predict(test_data)
print(pred)

import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
import tensorflow_docs.plots

# plotter = tfdocs.plots.HistoryPlotter(metrics= 'accuracy', smoothing_std  = 10)
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
# print((prediction[50]))
# print("prediction should be: " , test_label[50])
points = 0 
# predd = []

preed = []
for i in range(len(pred)):
	print(int(np.argmax(pred[i])),int(test_label[i]))
	preed.append(int(np.argmax(pred[i])))
	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
		points+=1

print("Accuracy : ", ( (points/ len(pred)) * 100))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.array(preed), test_label)
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation = 45)
plt.yticks(tick_marks,class_names)

import itertools
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j,i, format(cm[i,j],'d'), horizontalalignment = 'center', color = 'white' if cm[i,j] > (cm.max() / 2) else 'black')
	plt.tight_layout()
	plt.ylabel('true label')
	plt.xlabel('predicted label')
	plt.show()
	plt.close()
#'''