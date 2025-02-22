from scipy.io import wavfile
import scipy
from scipy.fftpack import dct

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

import librosa
from python_speech_features import mfcc, logfbank
import pandas as pd

import glob
import numpy as np
import os

from tqdm import tqdm
import copy
import time
import itertools


import sklearn

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


class MaskDetection():
    TRAIN_DATA_PATH = './ml-fmi-23-2020/train/train'
    VALIDATION_DATA_PATH = './ml-fmi-23-2020/validation/validation'
    TEST_DATA_PATH = './ml-fmi-23-2020/test/test'

    CLEAN_TRAIN_DATA_PATH = './clean_data/train/train'
    CLEAN_VALIDATION_DATA_PATH = './clean_data/validation/validation'
    CLEAN_TEST_DATA_PATH = './clean_data/test/test'

    CSV_FILES_FOLDER_PATH = './csv_feature_files'
    CLEAN_CSV_FILES_FOLDER_PATH = './csv_feature_files_clean'

    TRAIN_FEATURES_PATH = '/train_features.csv'
    VALIDATION_FEATURES_PATH = '/validation_features.csv'
    TEST_FEATURES_PATH = '/test_features.csv'

    TRAIN_MFCC_PATH = '/train_mean_mfcc.csv'
    VALIDATION_MFCC_PATH = '/validation_mean_mfcc.csv'
    TEST_MFCC_PATH = '/test_mean_mfcc.csv'

    TRAIN_LABELS_PATH = './ml-fmi-23-2020/train.txt'
    VALIDATION_LABELS_PATH = './ml-fmi-23-2020/validation.txt'

    OUTPUT_FILE_PATH = './submission_neural.txt'
    LOG_FILE_PATH = './logs/logHistory.txt'

    DIFFERENT_MODELS_PATH = './logs/differentModels.txt'
    ALL_FEATURE_COMBINATIONS_PATH = './logs/test_all_feature_combinations_svc.txt'

    SVC_HYPERPARAMETER_TUNING = './logs/svcHyperparameterTuning.txt'
    NEURAL_HYPERPARAMETER_TUNING = './logs/neuralHyperparameterTuning.txt'

    def __init__(self):
        self.train_names = []
        self.test_names = []
        self.validation_names = []

        self.train_data = []
        self.test_data = []
        self.validation_data = []

        self.train_labels = []
        self.validation_labels = []

        self.train_features = []
        self.validation_features = []
        self.test_features = []

        self.accuracy = 'default'
        self.recall = 'default'
        self.precision = 'default'
        self.runningTime = 0

        self.sr = 44100

    def envelope(self, signal, rate, threshold):
        # create a mask with True/False values which will be used
        # to reduce the empty/under the threshold portions of data
        mask = []
        y = copy.deepcopy(signal)
        # convert the numpy array into a series and transform each value in the series into it's absolute value
        y = pd.Series(y).apply(np.abs)
        # create a rolling window over the signal with pandas which provides rolling window calculations
        # and get the mean of the window
        y_mean = y.rolling(window=int(rate/10),
                           min_periods=1, center=True).mean()
        # create the True/False mask based on the threshold
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def cleanData(self):
        # if the folders in which the clean files are stored are empty, then the files are cleaned
        # the path to the data is changed to indicate to the cleaned files
        # the function is going to use a True/False mask to keep only the relevant parts of the signal
        # and write the clean files into the specified folder

        threshold = 0.0005

        if len(os.listdir(self.TRAIN_DATA_PATH)) == 0:
            # clean train data
            for filepath in tqdm(glob.glob(self.TRAIN_DATA_PATH + '/*')):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, threshold)
                wavfile.write(self.TRAIN_DATA_PATH + name,
                              rate=sr, data=data[mask])

            # clean validation data
            for filepath in tqdm(glob.glob(self.VALIDATION_DATA_PATH + '/*')):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, threshold)
                wavfile.write(self.VALIDATION_DATA_PATH +
                              name, rate=sr, data=data[mask])

            # clean test data
            for filepath in tqdm(glob.glob(self.TEST_DATA_PATH + '/*')):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, threshold)
                wavfile.write(self.TEST_DATA_PATH + name,
                              rate=sr, data=data[mask])
        else:
            print('\nAttention: Data files have been cleaned before. If you want to clean them again, try removing old folder files and then clean again.')

        # once the data has been cleaned, we changed the paths to the data to point to the clean ones
        self.TRAIN_DATA_PATH = self.CLEAN_TRAIN_DATA_PATH
        self.VALIDATION_DATA_PATH = self.CLEAN_VALIDATION_DATA_PATH
        self.TEST_DATA_PATH = self.CLEAN_TEST_DATA_PATH
        self.CSV_FILES_FOLDER_PATH = self.CLEAN_CSV_FILES_FOLDER_PATH

    def get_fft(self, y, rate):
        n = len(y)
        frequency = np.fft.rfftfreq(n, d=1/rate)
        magnitude = abs(np.fft.rfft(y)/n)
        return (magnitude, frequency)

    def extractFeaturesForDataSet(self, set_of_data, names_for_data, filename):
        # create the header for the csv file
        to_append = f'filename,mean_spectral_centroids,mean_spectral_rolloff,mean_spectral_bandwidth_2,sum_zero_crossings,mean_mfccs,'
        to_append += f'mean_magnitude,mean_freq,mean_chroma_stft,mean_rms,mean_bank,mean_mel,mean_spectral_contrast,mean_chroma_med,mean_melspectrogram,mean_flatness'
        g = open(filename, 'w')
        g.write(to_append)
        g.close()

        g = open(filename, 'a')

        for index in tqdm(range(len(set_of_data))):
            data = set_of_data[index]

            spectral_centroids = librosa.feature.spectral_centroid(
                y=data, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=data, sr=self.sr)
            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(
                y=data, sr=self.sr)

            zero_crossings = librosa.zero_crossings(data)
            mfccs = librosa.feature.mfcc(y=data, sr=self.sr)

            chroma_stft = librosa.feature.chroma_stft(y=data, sr=self.sr)
            rms = librosa.feature.rms(y=data)

            bank = logfbank(data[:self.sr], self.sr, nfilt=26, nfft=1103)
            mel = mfcc(data, self.sr, numcep=13, nfilt=26, nfft=1103)

            spectral_contrast = librosa.feature.spectral_contrast(
                data, sr=self.sr)

            magnitude, freq = self.get_fft(data, self.sr)

            chroma_cqt = librosa.feature.chroma_cqt(y=data, sr=self.sr)
            chroma_med = librosa.decompose.nn_filter(
                chroma_cqt, aggregate=np.median, metric='cosine')

            melspectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)

            flatness = librosa.feature.spectral_flatness(y=data)

            # calculate the mean or sum for the extracted features
            mean_spectral_centroids = np.mean(spectral_centroids)
            mean_spectral_rolloff = np.mean(spectral_rolloff)
            mean_spectral_bandwidth_2 = np.mean(spectral_bandwidth_2)
            sum_zero_crossings = sum(zero_crossings)
            mean_mfccs = np.mean(mfccs)

            mean_magnitude = np.mean(magnitude)
            mean_freq = np.mean(freq)
            mean_chroma_stft = np.mean(chroma_stft)
            mean_rms = np.mean(rms)
            mean_bank = np.mean(bank)
            mean_mel = np.mean(mel)

            mean_spectral_contrast = np.mean(spectral_contrast)
            mean_chroma_med = np.mean(chroma_med)
            mean_melspectrogram = np.mean(melspectrogram)
            mean_flatness = np.mean(flatness)

            to_append = f'\n{names_for_data[index]},{mean_spectral_centroids},{mean_spectral_rolloff},{mean_spectral_bandwidth_2},{sum_zero_crossings},{mean_mfccs},'
            to_append += f'{mean_magnitude},{mean_freq},{mean_chroma_stft},{mean_rms},{mean_bank},{mean_mel},{mean_spectral_contrast},{mean_chroma_med},'
            to_append += f'{mean_melspectrogram},{mean_flatness}'
            g.write(to_append)

        g.close()

    def extractFeaturesAllData(self):
        # if the files which should contain the features for the training data are empty,
        # then the files are created and the features are extracted and stored in them

        self.TRAIN_FEATURES_PATH = self.CSV_FILES_FOLDER_PATH + self.TRAIN_FEATURES_PATH
        self.VALIDATION_FEATURES_PATH = self.CSV_FILES_FOLDER_PATH + \
            self.VALIDATION_FEATURES_PATH
        self.TEST_FEATURES_PATH = self.CSV_FILES_FOLDER_PATH + self.TEST_FEATURES_PATH

        if os.path.exists(self.TRAIN_FEATURES_PATH) == False:
            self.readData()

            print('\nextracting features for train, validation and test data')
            self.extractFeaturesForDataSet(
                self.train_data, self.train_names, self.TRAIN_FEATURES_PATH)
            self.extractFeaturesForDataSet(
                self.validation_data, self.validation_names, self.VALIDATION_FEATURES_PATH)
            self.extractFeaturesForDataSet(
                self.test_data, self.test_names, self.TEST_FEATURES_PATH)
        else:
            print('\nAttention: Features have been extracted before. If you want to extract them again, try removing old csv files and then extract again.')

    def extractMeanMfccsForDataSet(self, set_of_data, names_for_data, filename):
        set_n_mfcc = 40
        set_n_fft = 2048
        set_hop_length = 512

        to_append = 'filename'
        for i in range(set_n_mfcc):
            to_append += f',{str(i+1)}'

        g = open(filename, 'w')
        g.write(to_append)
        g.close()

        g = open(filename, 'a')

        for index in tqdm(range(len(set_of_data))):
            data = set_of_data[index]

            mfccs = librosa.feature.mfcc(
                y=data, sr=self.sr, n_fft=set_n_fft, hop_length=set_hop_length, n_mfcc=set_n_mfcc)
            mfccs = np.mean(mfccs.T, axis=0)
            to_append = f'\n{names_for_data[index]}'
            for ind in range(len(mfccs)):
                to_append += f',{np.mean(mfccs[ind])}'

            g.write(to_append)

        g.close()

    def extractMeanMfccsAllData(self):
        # if the files which should contain the mfcc values for the training data are empty,
        # then the files are created and the mfcc values are extracted and stored in them

        self.TRAIN_MFCC_PATH = self.CSV_FILES_FOLDER_PATH + self.TRAIN_MFCC_PATH
        self.VALIDATION_MFCC_PATH = self.CSV_FILES_FOLDER_PATH + self.VALIDATION_MFCC_PATH
        self.TEST_MFCC_PATH = self.CSV_FILES_FOLDER_PATH + self.TEST_MFCC_PATH

        if os.path.exists(self.TRAIN_MFCC_PATH) == False:
            self.readData()

            print('\nextracting features for train, validation and test data')
            self.extractMeanMfccsForDataSet(
                self.train_data, self.train_names, self.TRAIN_MFCC_PATH)
            self.extractMeanMfccsForDataSet(
                self.validation_data, self.validation_names, self.VALIDATION_MFCC_PATH)
            self.extractMeanMfccsForDataSet(
                self.test_data, self.test_names, self.TEST_MFCC_PATH)
        else:
            print('\nAttention: Mean mfccs have been extracted before. If you want to extract them again, try removing old csv files and then extract again.')

    def loadFeaturesForDataSet(self, filename, data_names):
        data_features = [0] * len(data_names)
        fd = open(filename, 'r')

        # we jump over the first line which contains the names of the fields
        for line in tqdm(fd.readlines()[1:]):
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = [float(elem) for elem in features[1:]]
                data_features[data_names.index(name)] = numeric_data_features

        fd.close()
        return data_features

    def loadFeaturesAllData(self):
        print('\nloading features for train, validation and test data')
        self.train_features = pb.loadFeaturesForDataSet(
            self.TRAIN_FEATURES_PATH, self.train_names)
        self.validation_features = pb.loadFeaturesForDataSet(
            self.VALIDATION_FEATURES_PATH, self.validation_names)
        self.test_features = pb.loadFeaturesForDataSet(
            self.TEST_FEATURES_PATH, self.test_names)

    def loadMeanMfccForDataSet(self, filename, data_names):
        data_features = [0] * len(data_names)
        fd = open(filename, 'r')

        # we jump over the first line which contains the names of the fields
        for line in tqdm(fd.readlines()[1:]):
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = [float(elem) for elem in features[1:]]
                data_features[data_names.index(name)] = numeric_data_features

        fd.close()
        return data_features

    def loadMeanMfccAllData(self):
        print('\nloading mean mfcc for train, validation and test data')
        self.train_features = pb.loadMeanMfccForDataSet(
            self.TRAIN_MFCC_PATH, self.train_names)
        self.validation_features = pb.loadMeanMfccForDataSet(
            self.VALIDATION_MFCC_PATH, self.validation_names)
        self.test_features = pb.loadMeanMfccForDataSet(
            self.TEST_MFCC_PATH, self.test_names)

    def loadMeanAndFeaturesForDataSet(self, filename_features, filename_mean, data_names):
        data_features = [0] * len(data_names)

        fd = open(filename_features, 'r')
        # we jump over the first line which contains the names of the fields
        for line in tqdm(fd.readlines()[1:]):
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = [float(elem) for elem in features[1:]]
                data_features[data_names.index(name)] = numeric_data_features

        fd.close()

        fd = open(filename_mean, 'r')
        # we jump over the first line which contains the names of the fields
        for line in tqdm(fd.readlines()[1:]):
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = [float(elem) for elem in features[1:]]
                data_features[data_names.index(name)].extend(
                    numeric_data_features)

        fd.close()
        return data_features

    def loadMeanAndFeaturesAllData(self):
        print('\nloading mean mfcc and features for train, validation and test data')
        self.train_features = pb.loadMeanAndFeaturesForDataSet(self.TRAIN_FEATURES_PATH,
                                                               self.TRAIN_MFCC_PATH, self.train_names)
        self.validation_features = pb.loadMeanAndFeaturesForDataSet(self.VALIDATION_FEATURES_PATH,
                                                                    self.VALIDATION_MFCC_PATH, self.validation_names)
        self.test_features = pb.loadMeanAndFeaturesForDataSet(self.TEST_FEATURES_PATH,
                                                              self.TEST_MFCC_PATH, self.test_names)

    def readData(self):
        print('\nreading train, validation and test data')
        self.train_data = []
        self.test_data = []
        self.validation_data = []

        self.train_names = []
        self.test_names = []
        self.validation_names = []

        # read train data
        for filepath in tqdm(glob.glob(self.TRAIN_DATA_PATH + '/*')):
            data, sr = librosa.load(filepath, sr=self.sr)
            self.sr = sr
            self.train_names.append(os.path.basename(filepath))
            self.train_data.append(data)

        self.train_data = np.array(self.train_data)

        # read validation data
        for filepath in tqdm(glob.glob(self.VALIDATION_DATA_PATH + '/*')):
            data, sr = librosa.load(filepath, sr=self.sr)
            self.validation_names.append(os.path.basename(filepath))
            self.validation_data.append(data)

        self.validation_data = np.array(self.validation_data)

        # read test data
        for filepath in tqdm(glob.glob(self.TEST_DATA_PATH + '/*')):
            data, sr = librosa.load(filepath, sr=self.sr)
            self.test_names.append(os.path.basename(filepath))
            self.test_data.append(data)

        self.test_data = np.array(self.test_data)

    def readLabels(self):
        print('\nloading labels for train, validation and test data')
        # read train labels
        fd = open(self.TRAIN_LABELS_PATH, 'r')

        self.train_labels = [0] * len(self.train_data)
        for line in tqdm(fd.readlines()):
            name = line.split(',')[0]
            if name in self.train_names:
                self.train_labels[self.train_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.train_labels = np.array(self.train_labels)

        # read validation labels
        fd = open(self.VALIDATION_LABELS_PATH, 'r')

        self.validation_labels = [0] * len(self.validation_data)
        for line in tqdm(fd.readlines()):
            name = line.split(',')[0]
            if name in self.validation_names:
                self.validation_labels[self.validation_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.validation_labels = np.array(self.validation_labels)

    def normalizationStandardizationLab(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.train_features)
        self.train_features = scaler.transform(self.train_features)
        self.validation_features = scaler.transform(self.validation_features)
        self.test_features = scaler.transform(self.test_features)

    def standardizationScale(self):
        self.train_features = preprocessing.scale(self.train_features)
        self.validation_features = preprocessing.scale(
            self.validation_features)
        self.test_features = preprocessing.scale(self.test_features)

    def normalizationNormalize(self):
        self.train_features = preprocessing.normalize(self.train_features)
        self.validation_features = preprocessing.normalize(
            self.validation_features)
        self.test_features = preprocessing.normalize(self.test_features)

    def svcAlgorithm(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()
        self.standardizationScale()

        model = SVC(probability=True, C=5, gamma=0.001, kernel='rbf')

        print("\nfit train features... ")
        model.fit(self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict validation features... ")
        predictions = model.predict(self.validation_features)
        print("predict validation features... done")

        # plot precision-recall curve and confusion matrix
        prob = model.predict_proba(self.validation_features)
        skplt.metrics.plot_precision_recall_curve(self.validation_labels, prob)
        skplt.metrics.plot_confusion_matrix(
            self.validation_labels, predictions)
        plt.show()

        # calculate recall, precision and accuracy
        self.recall = round(recall_score(
            self.validation_labels, predictions), 3)
        self.precision = round(average_precision_score(
            self.validation_labels, predictions), 3)
        self.accuracy = np.mean(predictions == self.validation_labels)

        # calculate recall, precision and accuracy
        target_names = ['class 0', 'class 1']
        print(classification_report(self.validation_labels, predictions, target_names=target_names))

        print("predict test features... ")
        predictions = model.predict(self.test_features)
        print("predict test features... done")

        # create the submission file with the .wav file name and the predicted value
        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        # calculate the amount of time the algorithm has been running for
        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def svcAlgorithmWithValidationDataAsTrainData(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()

        self.train_features = np.append(
            self.train_features, self.validation_features, 0)
        self.train_labels = np.append(
            self.train_labels, self.validation_labels, 0)

        self.standardizationScale()

        model = SVC(probability=True, C=5, gamma=0.001,
                    kernel='rbf', degree=4)

        print("\nfit train features... ")
        model.fit(self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict test features... ")
        predictions = model.predict(self.test_features)
        print("predict test features... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def svcHyperparameterTuning(self):
        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()

        # try many parameters to find the best ones
        clf = GridSearchCV(SVC(gamma='auto'), {
            'C': [1, 5, 10, 15, 20, 25, 30, 35],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001, 1, 5],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }, cv=5, return_train_score=False)

        clf.fit(self.train_features, self.train_labels)
        data_frame = pd.DataFrame(clf.cv_results_)

        g = open(self.SVC_HYPERPARAMETER_TUNING, 'a')
        g.write('\n\n\nNew hyperparameter tuning\n')
        g.write(
            str(data_frame[['param_C', 'param_kernel', 'mean_test_score']]))
        g.write(f'\nBest params: {clf.best_params_}')
        g.write(f'\nBest score: {clf.best_score_}')
        g.close()

    def neuralAlgorithm(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()
        self.standardizationScale()

        print("\nfit train features... ")
        clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=1, max_iter=200, early_stopping=False).fit(
            self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict validation features... ")
        predictions = clf.predict(self.validation_features)
        print("predict validation features... done")

        # plot precision-recall curve and confusion matrix
        prob = clf.predict_proba(self.validation_features)
        skplt.metrics.plot_precision_recall_curve(self.validation_labels, prob)
        skplt.metrics.plot_confusion_matrix(
            self.validation_labels, predictions)
        plt.show()

        # calculate recall, precision and accuracy
        self.recall = round(recall_score(
            self.validation_labels, predictions), 3)
        self.precision = round(average_precision_score(
            self.validation_labels, predictions), 3)
        self.accuracy = clf.score(
            self.validation_features, self.validation_labels)
        
        # calculate recall, precision and accuracy
        target_names = ['class 0', 'class 1']
        print(classification_report(self.validation_labels, predictions, target_names=target_names))

        print("predict test features... ")
        predictions = clf.predict(self.test_features)
        print("predict test features... done")

        # create the submission file with the .wav file name and the predicted value
        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        # calculate the amount of time the algorithm has been running for
        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def neuralAlgorithmWithValidationDataAsTrainData(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()

        self.train_features = np.append(
            self.train_features, self.validation_features, 0)
        self.train_labels = np.append(
            self.train_labels, self.validation_labels, 0)

        self.standardizationScale()

        print("\nfit train features... ")
        clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=1, max_iter=200, early_stopping=False).fit(
            self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict test features... ")
        predictions = clf.predict(self.test_features)
        print("predict test features... done")

        # create the submission file with the .wav file name and the predicted value
        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        # calculate the amount of time the algorithm has been running for
        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def neuralHyperparameterTuning(self):
        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()

        #  try many parameters to find the best ones
        clf = GridSearchCV(MLPClassifier(), {
            'activation': ['relu'],
            'solver': ['adam', 'sgd'],
            'max_iter': [20, 50, 100, 150, 200, 250, 300, 350, 400]
        }, cv=5, return_train_score=False)
        clf.fit(self.train_features, self.train_labels)
        data_frame = pd.DataFrame(clf.cv_results_)

        g = open(self.NEURAL_HYPERPARAMETER_TUNING, 'a')
        g.write('\n\n\nNew hyperparameter tuning\n')
        g.write(str(data_frame[[
                'param_activation', 'param_solver', 'param_max_iter', 'mean_test_score']]))
        g.write(f'\nBest params: {clf.best_params_}')
        g.write(f'\nBest score: {clf.best_score_}')
        g.close()

    def knnAlgorithm(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadFeaturesAllData()

        # # normalizare/standardizare
        # # facem statisticile pe datele de antrenare
        # scaler = preprocessing.StandardScaler()
        # scaler.fit(self.train_features)
        # # scalam datele de antrenare
        # self.train_features = scaler.transform(self.train_features)
        # # scalam datele de validare
        # self.validation_features = scaler.transform(self.validation_features)
        # # scalam datele de test
        # self.test_features = scaler.transform(self.test_features)

        self.train_features = preprocessing.scale(self.train_features)
        self.validation_features = preprocessing.scale(
            self.validation_features)
        self.test_features = preprocessing.scale(self.test_features)

        print("\nfit train features... ")
        clf = KNeighborsClassifier(n_neighbors=20).fit(
            self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict validation features... ")
        predictions = clf.predict(self.validation_features)
        print("predict validation features... done")

        self.accuracy = clf.score(
            self.validation_features, self.validation_labels)

        print("predict test features... ")
        predictions = clf.predict(self.test_features)
        print("predict test features... done")

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

        # #  try many parameters to find the best ones
        # clf = GridSearchCV(KNeighborsClassifier(), {
        #     'n_neighbors': [1,5,10,15,20,25,30],
        #     'weights': ['uniform', 'distance'],
        #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        # }, cv=5, return_train_score=False)
        # clf.fit(self.train_features, self.train_labels)
        # df = pd.DataFrame(clf.cv_results_)
        # print(df[['param_n_neighbors', 'param_weights',
        #           'param_algorithm', 'mean_test_score']])
        # print("\n\n")
        # print(clf.best_params_)
        # print(clf.best_score_)

    def kerasNeuralNetwork(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadMeanAndFeaturesAllData()
        self.standardizationScale()

        # create model, add dense layers one by one specifying activation function
        # Units are the dimensionality of the output space for the layer, which equals the number of hidden units
        model = Sequential()
        
        # input layer requires input_dim param
        model.add(Dense(units=110, input_dim=55, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(units=70, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(units=20, activation='relu'))
        model.add(Dropout(.2))
        # sigmoid instead of relu for final probability between 0 and 1
        model.add(Dense(units=1, activation='sigmoid'))

        # The compile method configures the model’s learning process
        model.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=['accuracy'])

        # The fit method does the training in batches
        # validation_features and validation_labels are Numpy arrays --just like in the Scikit-Learn API.
        model.fit(self.train_features, self.train_labels,
                  epochs=200, batch_size=50)

        # The evaluate method calculates the losses and metrics for the trained model
        # Evaluating the model on the training set
        score = model.evaluate(self.train_features,
                               self.train_labels, verbose=0)
        print("Training Accuracy: {0:.3%}".format(score[1]))

        # Evaluating the model on the validation set
        score = model.evaluate(self.validation_features,
                               self.validation_labels, verbose=0)
        print("Validation Accuracy: {0:.3%}".format(score[1]))

        # Predict labels for validation data set
        predictions = model.predict_classes(
            self.validation_features, verbose=0)
        predictions = [x[0] for x in predictions]

        # calculate recall, precision and accuracy
        self.recall = round(recall_score(
            self.validation_labels, predictions), 3)
        self.precision = round(average_precision_score(
            self.validation_labels, predictions), 3)
        self.accuracy = np.mean(predictions == self.validation_labels)

        # calculate recall, precision and accuracy
        target_names = ['class 0', 'class 1']
        print(classification_report(self.validation_labels, predictions, target_names=target_names))

        # Predict labels for test data set
        predictions = model.predict_classes(self.test_features, verbose=0)
        predictions = [x[0] for x in predictions]

        g = open(self.OUTPUT_FILE_PATH, 'w')
        g.write('name,label')
        for index in range(len(self.test_names)):
            g.write(f'\n{self.test_names[index]},{predictions[index]}')
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def showRunningLogs(self):
        print(f'\nAccuracy: {self.accuracy}')
        print(f'Precision: {self.precision}')
        print(f'Recall: {self.recall}')
        print(f'Running time: {self.runningTime}min')

        g = open(self.LOG_FILE_PATH, 'a')
        localtime = time.asctime(time.localtime(time.time()))
        g.write(
            localtime + f'  accuracy: {self.accuracy}, precision: {self.precision}, recall: {self.recall}, running time: {self.runningTime}\n\n')
        g.close()

    def tryDifferentModels(self):
        start_time = time.time()

        self.readData()
        self.readLabels()
        self.loadFeaturesAllData()

        # self.normalizationStandardizationLab()
        self.standardizationScale()

        model_params = {
            'neuronal': {
                'model': MLPClassifier(),
                'params': {
                    'activation': ['relu']
                    # ,
                    # 'kernel': ['rbf', 'linear']
                }
            }
            # 'svm': {
            #     'model': SVC(gamma='auto'),
            #     'params': {
            #         'C': [1, 10, 20],
            #         'kernel': ['rbf', 'linear']
            #     }
            # },
            # 'random_forest': {
            #     'model': RandomForestClassifier(),
            #     'params': {
            #         'n_estimators': [1, 5, 10]
            #     }
            # },
            # 'logistic_regression': {
            #     'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            #     'params': {
            #         'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            #         'C': np.logspace(-4, 4, 20),
            #         'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
            #         'max_iter': [100, 1000, 2500, 5000]
            #     }
            # },
            # 'naive_bayes_gaussian': {
            #     'model': GaussianNB(),
            #     'params': {}
            # },
            # 'naive_bayes_multinomial': {
            #     'model': MultinomialNB(),
            #     'params': {}
            # },
            # 'decision_tree': {
            #     'model': DecisionTreeClassifier(),
            #     'params': {
            #         'criterion': ['gini', 'entropy'],
            #     }
            # }
        }

        scores = []

        for model_name, mp in model_params.items():
            clf = GridSearchCV(mp['model'], mp['params'],
                               cv=5, return_train_score=False)
            clf.fit(self.train_features, self.train_labels)
            scores.append({
                'model': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_
            })

        df = pd.DataFrame(
            scores, columns=['model', 'best_score', 'best_params'])
        print(df)

        g = open(self.DIFFERENT_MODELS_PATH, 'a')
        g.write('\n\n' + str(df))
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def loadCertainFeaturesForDataSet(self, filename, data_names, list_index):
        data_features = [0] * len(data_names)
        fd = open(filename, 'r')

        # we jump over the first line which contains the names of the fields
        for line in fd.readlines()[1:]:
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = []
                for index in list_index:
                    numeric_data_features.append(float(features[index]))
                data_features[data_names.index(name)] = numeric_data_features

        fd.close()
        return data_features

    def loadCertainFeaturesAllData(self, list_index):
        self.train_features = pb.loadCertainFeaturesForDataSet(
            self.TRAIN_FEATURES_PATH, self.train_names, list_index)
        self.validation_features = pb.loadCertainFeaturesForDataSet(
            self.VALIDATION_FEATURES_PATH, self.validation_names, list_index)
        self.test_features = pb.loadCertainFeaturesForDataSet(
            self.TEST_FEATURES_PATH, self.test_names, list_index)

    def tryAllFeatureCombinationsForSvc(self):
        start_time = time.time()

        g = open(self.ALL_FEATURE_COMBINATIONS_PATH, 'a')
        localtime = time.asctime(time.localtime(time.time()))
        g.write(f'\n\n{localtime} New test session\n')
        g.close()

        self.readData()
        self.readLabels()

        feature_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        best_accuracy = 0
        best_subset = []
        for L in range(1, len(feature_indexes)+1):
            for subset in itertools.combinations(feature_indexes, L):
                subset = list(subset)
                print(subset)

                self.loadCertainFeaturesAllData(subset)

                # self.normalizationStandardizationLab()
                self.standardizationScale()

                model = SVC(C=5, gamma=0.001, kernel='rbf')
                model.fit(self.train_features, self.train_labels)

                predictions = model.predict(self.validation_features)

                self.accuracy = np.mean(predictions == self.validation_labels)

                stop_time = time.time()
                self.runningTime = round(int(stop_time - start_time)/60, 2)

                if(self.accuracy > best_accuracy):
                    best_accuracy = self.accuracy
                    best_subset = subset

                g = open(self.ALL_FEATURE_COMBINATIONS_PATH, 'a')
                localtime = time.asctime(time.localtime(time.time()))
                g.write(
                    localtime + f'  accuracy: {self.accuracy}, running time: {self.runningTime}, subset: {subset}\n')
                g.close()

        print(best_accuracy)
        print(best_subset)


pb = MaskDetection()

# if the folders in which the clean files are stored are empty, then the files are cleaned
# the path to the training date is changed to indicate to the cleaned files
pb.cleanData()

# if the files which should contain the features for the training data are empty,
# then the files are created and the features are extracted and stored in them
pb.extractFeaturesAllData()

# if the files which should contain the mfcc values for the training data are empty,
# then the files are created and the mfcc values are extracted and stored in them
pb.extractMeanMfccsAllData()


# pb.svcAlgorithm()
# pb.svcAlgorithmWithValidationDataAsTrainData()

# pb.neuralAlgorithm()
# pb.neuralAlgorithmWithValidationDataAsTrainData()

pb.kerasNeuralNetwork()

# pb.knnAlgorithm()

pb.showRunningLogs()

# pb.svcHyperparameterTuning()
# pb.neuralHyperparameterTuning()

# pb.tryDifferentModels()
# pb.tryAllFeatureCombinationsForSvc()
