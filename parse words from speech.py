import numpy as np 
import matplotlib.pyplot as plt 
<<<<<<< HEAD

import os
import glob
from scipy.io import wavfile

audio_filename = "*/*/*/*.wav"
emg_filename = "*/*/*/*.adc"
transcript_filename = "*/*/*/transcript*.txt"
destination_folder = ['Extracted Words audio','Extracted Words emg'] 
=======
import pandas as pd 
import os
import glob
from scipy.io import wavfile
import librosa

audio_filename = "*/*/*/*.wav"
transcript_filename = "*/*/*/transcript*.txt"
destination_folder = "Extracted Words"
>>>>>>> d3b072844492aab99ccb7acc673748156017d3d3
offset_filename = "*/*/*/offset*.txt"
align_filename = "*/*/*/words*.txt"
main_dir = os.getcwd()


<<<<<<< HEAD

for destination_fold in destination_folder:
	if (not(destination_fold in os.listdir())):
		os.makedirs(destination_fold)
	else:
		print("The folder already exists")
=======
if (not(destination_folder in os.listdir())):
	os.makedirs(destination_folder)
else:
	print("The folder already exists")


class Parser:
	def __init__(self,):
		new_words = []
		counts = []
>>>>>>> d3b072844492aab99ccb7acc673748156017d3d3


def get_files(filename):
	files = []
	for file in glob.glob(filename):
		files.append(file)
	return files


def read_file(file):
	f = open(file,"r")
	data = f.read()
	data = data[0:-1].split("\n")
	f.close()
	return data

def trim_audio(audio_files,offset_files):
	audio = []
	for i in range(len(audio_files)):
		SR, a = wavfile.read(audio_files[i])
		a = a[:,0]
		trimmer = read_file(offset_files[i])[0].split(" ")
		a = a[int(trimmer[0]):int(trimmer[1])]
		audio.append(a)
	return audio,SR


<<<<<<< HEAD
def trim_emg(emg_files, offset_files):
	emg = []
	import sys
	import struct
	def ADC2arr(filename):
		raw = open(filename,"rb")
		signal = (raw.read())
		h_endian = 'h'*int(len(signal) / 2)
		values = list(struct.unpack('<'+h_endian,signal))
		values = np.array(values)
		values = values.reshape(int(len(signal) / (2*7)),7)
		values = values.T
		raw.close()
		return values			#rows -> #channel , Cols -> data points
	
	for i in range(len(emg_files)):
		a = ADC2arr(emg_files[i])
		a = a[:-1,:]											#trimming the marker signal
		trimmer = read_file(offset_files[i])[1].split(" ")
		a = a[:,int(trimmer[0]):int(trimmer[1])]
		emg.append(a)
	return emg

def extract_words(audio,audio_files,align_files,top_10):
=======
def extract_words(audio,audio_files,align_files,top_10):
	# os.mkdir("Top 20 Words")
	if (not("Top 20 Words" in os.listdir())):
		os.makedirs("Top 20 Words")	
	else:
		print("The folder already exists")

>>>>>>> d3b072844492aab99ccb7acc673748156017d3d3
	for j in range(len(audio)):
		temp_audio = audio[j]
		temp_name = audio_files[j][-16:]
		temp_align = read_file(align_files[j])
		for i in range(len(temp_align)):
			trimmer = temp_align[i].split(" ")
			try:
				w = temp_audio[int(trimmer[0])*160:int(trimmer[1])*160]
			except:
				print("Error at :",temp_name)
				print("ERROR : \n",temp_align)
				print("skipping that file ")
				continue
			
			try:
<<<<<<< HEAD
				wavfile.write(destination_folder[0]+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)		
			except:
				os.makedirs(destination_folder[0]+"/"+trimmer[2])
				wavfile.write(destination_folder[0]+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)
			
			try:
				if(trimmer[2] in top_10):
					wavfile.write("Top 10 Words audio"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)
				
			except:
				if(trimmer[2] in top_10):
					os.makedirs("Top 10 Words audio"+"/"+trimmer[2])	
					wavfile.write("Top 10 Words audio"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)

def extract_words_emg(emg, emg_files, align_files, top_10):
	for j in range(len(audio)):
		temp_emg = emg[j]
		temp_name = emg_files[j][-16:]
		temp_name = temp_name.replace('adc','csv')
		temp_align = read_file(align_files[j])
		for i in range(len(temp_align)):
			trimmer = temp_align[i].split(" ")
			try:
				w = temp_emg[ : , int(trimmer[0])*6:int(trimmer[1])*6]			#(sampleperframe) -->6  =  samplerate * 10**-3 * ms_per_frame(a frame of 10ms)
			except:
				print("Error at :",temp_name)
				print("ERROR : \n",temp_align)
				print("skipping that file ")
				continue

			import csv
			def emg_write(file_name, emg_data):
				with open(file_name, 'w') as file :
					writer = csv.writer(file, delimiter = ',')
					for row in emg_data:
						writer.writerow(row)
			try:
				emg_write(destination_folder[1]+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,w)		
			except:
				os.makedirs(destination_folder[1]+"/"+trimmer[2])
				emg_write(destination_folder[1]+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,w)
			
			try:
				if(trimmer[2] in top_10):
					emg_write("Top 10 Words emg"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,w)				
			except:
				if(trimmer[2] in top_10):
					os.makedirs("Top 10 Words emg"+"/"+trimmer[2])	
					emg_write("Top 10 Words emg"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,w)

audio_files = get_files(audio_filename)
emg_files = get_files(emg_filename)
offset_files = get_files(offset_filename)
align_files = get_files(align_filename)

audio_files.sort()
emg_files.sort()
offset_files.sort()
align_files.sort()

audio, SR = trim_audio(audio_files,offset_files)
emg = trim_emg(emg_files, offset_files)

=======
				wavfile.write(destination_folder+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)		
			except:
				os.makedirs(destination_folder+"/"+trimmer[2])
				wavfile.write(destination_folder+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)
			
			try:
				if(trimmer[2] in top_10):
					wavfile.write("Top 10 Words"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)
				
			except:
				if(trimmer[2] in top_10):
					os.makedirs("Top 10 Words"+"/"+trimmer[2])	
					wavfile.write("Top 10 Words"+"/"+trimmer[2]+"/"+trimmer[2]+"_"+temp_name,16000,w)


audio_files = get_files(audio_filename)
offset_files = get_files(offset_filename)
align_files = get_files(align_filename)
audio_files.sort()
offset_files.sort()
align_files.sort()
audio, SR = trim_audio(audio_files,offset_files)
>>>>>>> d3b072844492aab99ccb7acc673748156017d3d3

#just to verify the trimmed audio files
# wavfile.write("trimmed_audio_2.wav",16000,audio[2])
# wavfile.write("trimmed_audio_3.wav",16000,audio[3])
# wavfile.write("trimmed_audio_4.wav",16000,audio[4])

<<<<<<< HEAD
import pandas as pd
dataframe = pd.read_excel("Word count.xlsx")
print(dataframe.head(10))
import pandas as pd 
top_10 = list(dataframe.iloc[0:10,1])
# top_20 = list(dataframe.iloc[0:10,1])

extract_words(audio,audio_files,align_files,top_10)
extract_words_emg(emg,emg_files,align_files,top_10)
=======

dataframe = pd.read_excel("Word count.xlsx")

# print(dataframe.head(10))

top_10 = list(dataframe.iloc[0:10,1])
top_20 = list(dataframe.iloc[0:10,1])

extract_words(audio,audio_files,align_files,top_10)

# os.chdir(destination_folder)



# for each in os.listdir():
# 	if()
>>>>>>> d3b072844492aab99ccb7acc673748156017d3d3
