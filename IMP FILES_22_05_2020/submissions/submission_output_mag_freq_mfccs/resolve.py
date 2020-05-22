from scipy.io import wavfile
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import glob
import numpy as np
import os
import time
import librosa
from python_speech_features import mfcc, logfbank
import pandas as pd
# import noisereduce as nr


import sklearn


class Problem():
    TRAIN_DATA_PATH = './ml-fmi-23-2020/train/train/*'
    VALIDATION_DATA_PATH = './ml-fmi-23-2020/validation/validation/*'
    TEST_DATA_PATH = './ml-fmi-23-2020/test/test/*'
    TRAIN_LABELS_PATH = './ml-fmi-23-2020/train.txt'
    VALIDATION_LABELS_PATH = './ml-fmi-23-2020/validation.txt'
    OUTPUT_FILE_PATH = './submission_output_rename.txt'
    ACCURACY_FILE_PATH = './logHistory.txt'

    def __init__(self):
        self.train_names = []
        self.test_names = []
        self.validation_names = []

        self.train_data = []
        self.test_data = []
        self.validation_data = []

        self.train_labels = []
        self.validation_labels = []
        self.accuracy = 'can\'t be calculated'
        self.runningTime = 0

        self.sr = 44100
    
    def cleanData(self):
        if len(os.listdir('./clean_data/train/train')) == 0:
            path = './clean_data/train/train/'
            # clean train data
            for filepath in glob.glob(self.TRAIN_DATA_PATH):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])
            
            path = './clean_data/validation/validation/'
            # clean validation data
            for filepath in glob.glob(self.VALIDATION_DATA_PATH):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])

            path = './clean_data/test/test/'
            # clean test data
            for filepath in glob.glob(self.TEST_DATA_PATH):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])
            
        self.TRAIN_DATA_PATH = './clean_data/train/train/*'
        self.VALIDATION_DATA_PATH = './clean_data/validation/validation/*'
        self.TEST_DATA_PATH = './clean_data/test/test/*'

    def readData(self):
        # read train data
        for filepath in glob.glob(self.TRAIN_DATA_PATH):
            data, sr = librosa.load(filepath, sr=44100)
            self.sr = sr
            # data = librosa.decompose.nn_filter(data)
            # fs, data = wavfile.read(filepath)
            self.train_names.append(os.path.basename(filepath))
            self.train_data.append(data)

        self.train_data = np.array(self.train_data)

        # read validation data
        for filepath in glob.glob(self.VALIDATION_DATA_PATH):
            data, sr = librosa.load(filepath, sr=44100)
            # data = librosa.decompose.nn_filter(data)
            # fs, data = wavfile.read(filepath)
            self.validation_names.append(os.path.basename(filepath))
            self.validation_data.append(data)

        self.validation_data = np.array(self.validation_data)

        # read test data
        for filepath in glob.glob(self.TEST_DATA_PATH):
            data, sr = librosa.load(filepath, sr=44100)
            # data = librosa.decompose.nn_filter(data)
            # fs, data = wavfile.read(filepath)
            self.test_names.append(os.path.basename(filepath))
            self.test_data.append(data)

        self.test_data = np.array(self.test_data)

    def readLabels(self):
        # set train labels
        fd = open(self.TRAIN_LABELS_PATH, 'r')

        self.train_labels = [0] * len(self.train_data)
        for line in fd.readlines():
            name = line.split(',')[0]
            if name in self.train_names:
                self.train_labels[self.train_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.train_labels = np.array(self.train_labels)

        # set validation labels
        fd = open(self.VALIDATION_LABELS_PATH, 'r')

        self.validation_labels = [0] * len(self.validation_data)
        for line in fd.readlines():
            name = line.split(',')[0]
            if name in self.validation_names:
                self.validation_labels[self.validation_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.validation_labels = np.array(self.validation_labels)

    def readValidationDataAsTrainData(self):
        # TRAIN DATA INCLUDES VALIDATION DATA
        # read train data
        for filepath in glob.glob(self.TRAIN_DATA_PATH):
            fs, data = wavfile.read(filepath)
            self.train_names.append(os.path.basename(filepath))
            self.train_data.append(data)

        # read validation data and treat it as train data
        for filepath in glob.glob(self.VALIDATION_DATA_PATH):
            fs, data = wavfile.read(filepath)
            self.train_names.append(os.path.basename(filepath))
            self.train_data.append(data)

        self.train_data = np.array(self.train_data)

        # read test data
        for filepath in glob.glob(self.TEST_DATA_PATH):
            fs, data = wavfile.read(filepath)
            self.test_names.append(os.path.basename(filepath))
            self.test_data.append(data)

        self.test_data = np.array(self.test_data)

    def readValidationLabelsAsTrainLabels(self):
        # set train labels
        fd = open(self.TRAIN_LABELS_PATH)

        self.train_labels = [0] * len(self.train_data)
        for line in fd.readlines():
            name = line.split(',')[0]
            if name in self.train_names:
                self.train_labels[self.train_names.index(
                    name)] = (int(line.split(',')[1]))
        fd.close()

        # set validation labels
        fd = open(self.VALIDATION_LABELS_PATH)

        for line in fd.readlines():
            name = line.split(',')[0]
            if name in self.train_names:
                self.train_labels[self.train_names.index(name)] = (
                    int(line.split(',')[1]))

        fd.close()
        self.train_labels = np.array(self.train_labels)

    def svcAlgorithm(self):
        start_time = time.time()

        print('reading data... ')
        self.readData()
        self.readLabels()
        print('reading data... done')

        print('extracting features... ')
        self.changeDataToFeatures()
        print('extracting features... done')

        # # normalizare/standardizare
        # # facem statisticile pe datele de antrenare
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(self.train_data)
        # # scalam datele de antrenare
        # self.train_data = scaler.transform(self.train_data)
        # # scalam datele de validare
        # self.validation_data = scaler.transform(self.validation_data)
        # # scalam datele de test
        # self.test_data = scaler.transform(self.test_data)

        self.train_data = preprocessing.scale(self.train_data)
        self.validation_data = preprocessing.scale(self.validation_data)
        self.test_data = preprocessing.scale(self.test_data)

        model = SVC()
        print("fit train data... ")
        model.fit(self.train_data, self.train_labels)
        print("fit train data... done")
        print("predict validation data... ")
        predictions = model.predict(self.validation_data)
        print("predict validation data... done")

        self.accuracy = np.mean(predictions == self.validation_labels)

        print("predict test data... ")
        predictions = model.predict(self.test_data)
        print("predict test data... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def svcAlgorithmWithValidationDataAsTrainData(self):
        print('reading data... ')
        self.readValidationDataAsTrainData()
        self.readValidationLabelsAsTrainLabels()
        print('reading data... done')

        start_time = time.time()

        model = SVC()
        print("fit train data... ")
        model.fit(self.train_data, self.train_labels)
        print("fit train data... done")
        print("predict test data... ")
        predictions = model.predict(self.test_data)
        print("predict test data... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label\n')
        for index in range(len(self.test_names)):
            g.write(f'{self.test_names[index]},{predictions[index]}\n')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def neuralAlgorithm(self):
        start_time = time.time()

        print('reading data... ')
        self.readData()
        self.readLabels()
        print('reading data... done')

        print('extracting features... ')
        self.changeDataToFeatures()
        print('extracting features... done')

        # # normalizare/standardizare
        # # facem statisticile pe datele de antrenare
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(self.train_data)
        # # scalam datele de antrenare
        # self.train_data = scaler.transform(self.train_data)
        # # scalam datele de validare
        # self.validation_data = scaler.transform(self.validation_data)
        # # scalam datele de test
        # self.test_data = scaler.transform(self.test_data)

        print("fit train data... ")
        clf = MLPClassifier(random_state=1, max_iter=200, early_stopping=False).fit(
            self.train_data, self.train_labels)
        print("fit train data... done")

        print("predict validation data... ")
        predictions = clf.predict(self.validation_data)
        print("predict validation data... done")

        self.accuracy = clf.score(self.validation_data, self.validation_labels)

        print("predict test data... ")
        predictions = clf.predict(self.test_data)
        print("predict test data... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def neuralAlgorithmWithValidationDataAsTrainData(self):
        print('reading data... ')
        self.readValidationDataAsTrainData()
        self.readValidationLabelsAsTrainLabels()
        print('reading data... done')

        start_time = time.time()

        print("fit train data... ")
        clf = MLPClassifier(random_state=1, max_iter=300).fit(
            self.train_data, self.train_labels)
        print("fit train data... done")

        print("predict test data... ")
        predictions = clf.predict(self.test_data)
        print("predict test data... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label\n')
        for index in range(len(self.test_names)):
            g.write(f'{self.test_names[index]},{predictions[index]}\n')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def showRunningLogs(self):
        print(f'\nAccuracy: {self.accuracy}')
        print(f'Running time: {self.runningTime}min')

        g = open(self.ACCURACY_FILE_PATH, 'a')
        localtime = time.asctime(time.localtime(time.time()))
        g.write(
            localtime + f'  accuracy: {self.accuracy}, running time: {self.runningTime}\n')
        g.close()

    def reduceNoiseForData(self):
        # code for noise reduction on data
        print("ceva")

    def addNoiseToTrainData(self):
        for i in range(len(self.train_data)):
            noise = np.random.normal(0, 1, len(self.train_data[i]))
            self.train_data[i] = 0.8 * self.train_data[i] + 0.2 * noise

    def get_fft(self, y, rate):
        n = len(y)
        freq = np.fft.rfftfreq(n, d=1/rate)
        Y = abs(np.fft.rfft(y)/n)
        return [Y, freq]
    
    def envelope(self, y, rate, treshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window = int(rate/10), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > treshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def extractFeatures(self, set_of_data):
        new_set_of_data = []
        for data in set_of_data:
            magnitude, freq = self.get_fft(data, self.sr)
            # bank = logfbank(data[:self.sr], self.sr, nfilt=26, nfft=1103)
            # mel = mfcc(data, self.sr, numcep=13, nfilt=26, nfft=1103)

            # chroma_stft and rms decrease accuracy
            # chroma_stft = librosa.feature.chroma_stft(y=data, sr=self.sr)
            # rms = librosa.feature.rms(y=data)

            spectral_centroids = librosa.feature.spectral_centroid(
                y=data, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=data, sr=self.sr)

            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(
                y=data, sr=self.sr)
            # spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=data, sr=self.sr, p=3)
            # spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=data, sr=self.sr, p=4)

            # n0 = 9000
            # n1 = 9100
            # zero_crossings = librosa.zero_crossings(data[n0:n1], pad=False)
            zero_crossings = librosa.zero_crossings(data)
            sum(zero_crossings)
            # zcr = librosa.feature.zero_crossing_rate(data)

            mfccs = librosa.feature.mfcc(y=data, sr=self.sr)

            new_data = []
            # new_data.append(np.mean(rms))
            # new_data.append(np.mean(chroma_stft))
            # new_data.append(np.mean(mel))
            new_data.append(np.mean(spectral_centroids))
            new_data.append(np.mean(spectral_bandwidth_2))
            new_data.append(np.mean(spectral_rolloff))
            new_data.append(sum(zero_crossings)) 
            new_data.append(np.mean(mfccs))
            new_data.append(np.mean(magnitude))
            new_data.append(np.mean(freq))

            # to_append = f'{np.mean(spectral_centroids)} {np.mean(spectral_bandwidth_2)} {np.mean(spectral_rolloff)} {np.mean(zero_crossings)}'
            # for e in mfccs:
            #     # to_append += f' {np.mean(e)}'
                # new_data.append(np.mean(e))
 
            new_data = np.array(new_data)
            new_set_of_data.append(new_data)
        new_set_of_data = np.array(new_set_of_data)
        return new_set_of_data

    def changeDataToFeatures(self):
        self.train_data = self.extractFeatures(self.train_data)
        self.validation_data = self.extractFeatures(self.validation_data)
        self.test_data = self.extractFeatures(self.test_data)


pb = Problem()
pb.cleanData()
pb.svcAlgorithm()
pb.showRunningLogs()
