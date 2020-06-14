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



class Visulaize():
	"""docstring for Visulaize"""
	def __init__(self, arg):
		super(Visulaize, self).__init__()
		self.arg = arg

	def sepectrogram(word, sample_rate):
		frequency,time, spectrogram = signal.spectrogram(words[0], SR, window = 'hamming', nperseg = int(40/1000 * sample_rate), noverlap = 0)
		from scipy import signal
		f,t,zxx = signal.stft(words[0], SR, window = 'hamming', nperseg = int(40 / 1000 * SR), noverlap = 0)
		plt.pcolormesh(t,f,np.abs(zxx))
		plt.ylabel('frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()
		plt.close()

	def powerSpectrogram(word, sample_rate):
		import librosa
		import librosa.display
		s = np.abs(librosa.stft(word))
		print(librosa.power_to_db(s**2))
		plt.figure()
		plt.subplot(2,1,1)
		librosa.display.specshow(s**2, sr=sample_rate,y_axis='log')
		plt.colorbar()
		plt.title('power spectrogram')
		plt.subplot(2,1,2)
		librosa.display.specshow(librosa.power_to_db(s**2, ref=np.max), sr = SR, y_axis = 'log', x_axis = 'time')
		plt.colorbar(format='%+2.0f dB')
		plt.title('log power spectrogram')
		plt.tight_layout()
		plt.show()	
		plt.close()

	def melspectrogram(word, sample_rate, n_mels = 128):
		import librosa
		d = np.abs(librosa.stft(word))**2
		s = librosa.feature.melspectrogram(S=d,sr = sample_rate, n_mels = n_mels)
		plt.figure(figsize = (10, 4))
		s_db = librosa.power_to_db(s, ref = np.max)
		print(s_db)
		import librosa.display
		librosa.display.specshow(s_db, x_axis = 'time', y_axis= 'mel', sr = SR)
		plt.colorbar(format='%2.0f dB')
		plt.title('Mel-frequency spectrogram')
		plt.tight_layout()
		plt.show()
		plt.close()


# frequency,time, spectrogram = signal.spectrogram(words[0], SR, window = 'hamming', nperseg = int(40/1000 * sample_rate), noverlap = 0)
# from scipy import signal
# f,t,zxx = signal.stft(words[0], SR, window = 'hamming', nperseg = int(40 / 1000 * SR), noverlap = 0)
# plt.pcolormesh(t,f,np.abs(zxx))
# plt.ylabel('frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# #spectrum plots
# import librosa
# import librosa.display
# s = np.abs(librosa.stft(words[0]))
# print(librosa.power_to_db(s**2))
# plt.figure()
# plt.subplot(2,1,1)
# librosa.display.specshow(s**2, sr=SR,y_axis='log')
# plt.colorbar()
# plt.title('power spectrogram')
# plt.subplot(2,1,2)
# librosa.display.specshow(librosa.power_to_db(s**2, ref=np.max), sr = SR, y_axis = 'log', x_axis = 'time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('log power spectrogram')
# plt.tight_layout()
# plt.show()

# #melspectrogram
# import librosa
# d = np.abs(librosa.stft(words[0]))**2
# s = librosa.feature.melspectrogram(S=d,sr = SR, n_mels = 128)
# plt.figure(figsize = (10, 4))
# s_db = librosa.power_to_db(s, ref = np.max)
# print(s_db)
# import librosa.display
# librosa.display.specshow(s_db, x_axis = 'time', y_axis= 'mel', sr = SR)
# plt.colorbar(format='%2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()
# plt.close()

