import librosa
import numpy as np


class Features:
	def __init__(self,words,window_size,sampling_rate,len_max_array):
		self.words = words
		self.window_size = window_size
		self.SR = sampling_rate
		self.step_size = int((window_size/1000) * self.SR)
		self.num_windows = int(len_max_array / self.step_size)
		self.features = []
		self.temp_features = []
		self.spec_features = []
	
	def compute(self):
		self.temp_features = self.compute_temporal_features()
		self.spec_features = self.compute_spectral_features()

	def compute_temporal_features(self):
		print("Extracting temporal features")
		keep = 0
		for word in self.words:
			keep+=1
			if keep%100==0:
				print("-",end=" ")
			f = []
			for i in range(self.num_windows):
				seg = word[i*self.step_size:(i+1)*self.step_size]
				temp = [ self.RMS(seg) , self.Mean(seg), np.min(seg),np.std(seg),np.max(seg) ]
				f.append(np.array(temp))
			f.append(np.array(temp))
			f = np.nan_to_num(np.hstack(f))
			self.features.append(f)
		print("\n")
		return np.vstack(self.features)

	def compute_spectral_features(self):
		spec_features = []
		print('Extracting spectral features')
		keep = 0
		for word in self.words:
			keep += 1
			if keep % 100 == 0:
				print('-', end = ' ')
			spec_features.append(self.mean_mfcc(word))
		return spec_features

	def mean_mfcc(self, word, n_mfcc = 40):
		import librosa
		f_x = librosa.feature.mfcc(word,sr = self.SR, n_mfcc = n_mfcc)
		mfccssclaed = np.mean(f_x.T, axis = 0)
		return mfccssclaed

	def ZCR(self,seg):
		# seg = seg - self.Mean(seg)
		pos = seg>0
		npos = ~pos
		return len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0])

	def ARV(self,seg):
		seg= np.abs(seg)
		return np.mean(seg)

	def RMS(self,seg):
		return np.sqrt(np.mean(np.power(seg,2)))

	def Mean(self,seg):
		try:
			return np.mean(seg)
		except:
			return 0
