import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.stats import zscore
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

words, labels, SR, len_max_array = read_data()


print('no of files : ',len(words))
print('no of samples : ',len(words[0]))

sample_rate = 16000
win_size = 35 # 40 ms 
#using spectrogram
from scipy import signal
feature = []
for word in words:
	frequency, time, spectrogram = signal.spectrogram(word, sample_rate, window = 'hamming', nperseg = int(win_size/1000 * sample_rate), noverlap = 0)
	feature.append(spectrogram)


# tmp_labels = list(labels)
# counter = []
# for lab in np.unique(tmp_labels):
# 	counter.append(tmp_labels.count(lab))
# print(counter)
# mindatanumber = np.min(counter)
# new_feat= []
# new_label = []
# for indy in index:
# 	new_feat.extend(feature[indy:indy+mindatanumber])
# 	new_label.extend(labels[indy:indy+mindatanumber])

# feature = new_feat
# labels = new_label

# import random 
# mapindexpos = list(zip(feature, labels))
# random.shuffle(mapindexpos)
# feature, labels = zip(*mapindexpos)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(labels)

feature = np.array(feature)
labels = np.array(labels)
print(feature.shape)

feature = np.reshape(feature, (feature.shape[0], feature.shape[1], feature.shape[2], 1))
print(feature.shape)

from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(feature, labels, test_size = 0.10)

from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

clf = Sequential()
clf.add(Convolution2D(10,3,3, input_shape = (feature.shape[1], feature.shape[2],feature.shape[3]), activation = 'relu' ))
clf.add(MaxPooling2D(pool_size = (2, 2)))
clf.add(Flatten())
clf.add(Dense(output_dim = 64, activation = 'relu'))
clf.add(Dense(output_dim = 10, activation = 'sigmoid'))
clf.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
clf.summary()

history = clf.fit(train_data, train_label, validation_split = 0.2, epochs = 25, batch_size = 32)
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

points = 0 
for i in range(len(pred)):
	print(int(np.argmax(pred[i])),int(test_label[i]))
	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
		points+=1
print("Accuracy : ", ( (points/ len(pred)) * 100))

# plt.pcolormesh(time, frequency, spectrogram)
# plt.ylabel('frequency')
# plt.xlabel('time')
# plt.show()
#https://stackoverflow.com/questions/45100189/specify-number-of-samples-in-a-segment-for-short-time-fourier-transform-using-sc
#https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
#'''