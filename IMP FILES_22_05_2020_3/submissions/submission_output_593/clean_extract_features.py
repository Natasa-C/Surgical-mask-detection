from scipy.io import wavfile

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import librosa
from python_speech_features import mfcc, logfbank
import pandas as pd
# import noisereduce as nr

import glob
import numpy as np
import os

from tqdm import tqdm
import copy
import time
import itertools


import sklearn


class Problem():
    TRAIN_DATA_PATH = './ml-fmi-23-2020/train/train/*'
    VALIDATION_DATA_PATH = './ml-fmi-23-2020/validation/validation/*'
    TEST_DATA_PATH = './ml-fmi-23-2020/test/test/*'

    TRAIN_FEATURES_PATH = './clean_data/train_features.csv'
    VALIDATION_FEATURES_PATH = './clean_data/validation_features.csv'
    TEST_FEATURES_PATH = './clean_data/test_features.csv'

    TRAIN_LABELS_PATH = './ml-fmi-23-2020/train.txt'
    VALIDATION_LABELS_PATH = './ml-fmi-23-2020/validation.txt'

    OUTPUT_FILE_PATH = './submission_output_rename.txt'
    LOG_FILE_PATH = './logHistory.txt'

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

        self.accuracy = 'can\'t be calculated'
        self.runningTime = 0

        self.sr = 44100

    def envelope(self, signal, rate, treshold):
        mask = []
        y = copy.deepcopy(signal)
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10),
                           min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > treshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def cleanData(self):
        if len(os.listdir('./clean_data/train/train')) == 0:
            path = './clean_data/train/train/'
            # clean train data
            for filepath in tqdm(glob.glob(self.TRAIN_DATA_PATH)):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])

            path = './clean_data/validation/validation/'
            # clean validation data
            for filepath in tqdm(glob.glob(self.VALIDATION_DATA_PATH)):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])

            path = './clean_data/test/test/'
            # clean test data
            for filepath in tqdm(glob.glob(self.TEST_DATA_PATH)):
                name = os.path.basename(filepath)
                data, sr = librosa.load(filepath, sr=self.sr)
                mask = self.envelope(data, self.sr, 0.0005)
                wavfile.write(path + name, rate=sr, data=data[mask])
        else:
            print('\nAttention: Data files have been cleaned before. If you want to clean them again, try removing old folder files and then clean again.')

        self.TRAIN_DATA_PATH = './clean_data/train/train/*'
        self.VALIDATION_DATA_PATH = './clean_data/validation/validation/*'
        self.TEST_DATA_PATH = './clean_data/test/test/*'

    def get_fft(self, y, rate):
        n = len(y)
        frequency = np.fft.rfftfreq(n, d=1/rate)
        magnitude = abs(np.fft.rfft(y)/n)
        return (magnitude, frequency)

    def extractFeaturesForDataSet(self, set_of_data, names_for_data, filename):
        to_append = f'filename,mean_spectral_centroids,mean_spectral_rolloff,mean_spectral_bandwidth_2,sum_zero_crossings,mean_mfccs,'
        to_append += f'mean_magnitude,mean_freq,mean_chroma_stft,mean_rms,mean_bank,mean_mel'
        g = open(filename, 'w')
        g.write(to_append)
        g.close()

        g = open(filename, 'a')

        for index in tqdm(range(len(set_of_data))):
            data = set_of_data[index]

            # spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=data, sr=self.sr, p=3)
            # spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=data, sr=self.sr, p=4)
            # zcr = librosa.feature.zero_crossing_rate(data)
            # for e in mfccs:
            #     # to_append += f' {np.mean(e)}'

            spectral_centroids = librosa.feature.spectral_centroid(
                y=data, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=data, sr=self.sr)
            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(
                y=data, sr=self.sr)

            zero_crossings = librosa.zero_crossings(data)
            mfccs = librosa.feature.mfcc(y=data, sr=self.sr)

            # chroma_stft and rms decrease accuracy
            chroma_stft = librosa.feature.chroma_stft(y=data, sr=self.sr)
            rms = librosa.feature.rms(y=data)

            bank = logfbank(data[:self.sr], self.sr, nfilt=26, nfft=1103)
            mel = mfcc(data, self.sr, numcep=13, nfilt=26, nfft=1103)

            # magnitude, freq do not mean a lot
            magnitude, freq = self.get_fft(data, self.sr)

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

            to_append = f'\n{names_for_data[index]},{mean_spectral_centroids},{mean_spectral_rolloff},{mean_spectral_bandwidth_2},{sum_zero_crossings},{mean_mfccs},'
            to_append += f'{mean_magnitude},{mean_freq},{mean_chroma_stft},{mean_rms},{mean_bank},{mean_mel}'
            g.write(to_append)

        g.close()

    def extractFeaturesAllData(self):
        if os.path.exists('clean_data/train_features.csv') == False:
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

    def loadFeaturesForDataSet(self, filename, data_names):
        data_features = [0] * len(data_names)
        fd = open(filename, 'r')

        # we jump over the first line which contains the names of the fields
        for line in tqdm(fd.readlines()[1:]):
            features = line.split(',')
            name = features[0]
            if name in data_names:
                numeric_data_features = [float(elem) for elem in features[1:7]]
                numeric_data_features.append(float(features[9]))
                numeric_data_features.append(float(features[10]))
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

    def readData(self):
        print('\nreading train, validation and test data')
        self.train_data = []
        self.test_data = []
        self.validation_data = []

        self.train_names = []
        self.test_names = []
        self.validation_names = []

        # read train data
        for filepath in tqdm(glob.glob(self.TRAIN_DATA_PATH)):
            data, sr = librosa.load(filepath, sr=44100)
            self.sr = sr
            self.train_names.append(os.path.basename(filepath))
            self.train_data.append(data)

        self.train_data = np.array(self.train_data)

        # read validation data
        for filepath in tqdm(glob.glob(self.VALIDATION_DATA_PATH)):
            data, sr = librosa.load(filepath, sr=44100)
            self.validation_names.append(os.path.basename(filepath))
            self.validation_data.append(data)

        self.validation_data = np.array(self.validation_data)

        # read test data
        for filepath in tqdm(glob.glob(self.TEST_DATA_PATH)):
            data, sr = librosa.load(filepath, sr=44100)
            self.test_names.append(os.path.basename(filepath))
            self.test_data.append(data)

        self.test_data = np.array(self.test_data)

    def readLabels(self):
        print('\nloading labels for train, validation and test data')
        # set train labels
        fd = open(self.TRAIN_LABELS_PATH, 'r')

        self.train_labels = [0] * len(self.train_data)
        for line in tqdm(fd.readlines()):
            name = line.split(',')[0]
            if name in self.train_names:
                self.train_labels[self.train_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.train_labels = np.array(self.train_labels)

        # set validation labels
        fd = open(self.VALIDATION_LABELS_PATH, 'r')

        self.validation_labels = [0] * len(self.validation_data)
        for line in tqdm(fd.readlines()):
            name = line.split(',')[0]
            if name in self.validation_names:
                self.validation_labels[self.validation_names.index(
                    name)] = (int(line.split(',')[1]))

        fd.close()
        self.validation_labels = np.array(self.validation_labels)

    def svcAlgorithm(self):
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

        model = SVC(C=5, gamma='scale', kernel='rbf',
                    decision_function_shape='ovr')
        print("\nfit train features... ")
        model.fit(self.train_features, self.train_labels)
        print("fit train features... done")

        print("predict validation features... ")
        predictions = model.predict(self.validation_features)
        print("predict validation features... done")

        self.accuracy = np.mean(predictions == self.validation_labels)

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

        # try many parameters to find the best ones
        # clf = GridSearchCV(SVC(gamma='auto'), {
        #     'C': [1, 5, 10, 15, 20, 25, 30, 35],
        #     'gamma': ['scale', 'auto'],
        #     'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        # }, cv=5, return_train_score=False)
        # clf.fit(self.train_features, self.train_labels)
        # df = pd.DataFrame(clf.cv_results_)
        # print(df[['param_C', 'param_kernel', 'mean_test_score']])
        # print("\n\n")
        # print(clf.best_params_)
        # print(clf.best_score_)

    def neuralAlgorithm(self):
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
        clf = MLPClassifier(random_state=1, max_iter=200, early_stopping=False).fit(
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
        # clf = GridSearchCV(MLPClassifier(), {
        #     'activation' : ['relu'],
        #     'solver' : ['adam', 'sgd']
        # }, cv=5, return_train_score=False)
        # clf.fit(self.train_features, self.train_labels)
        # df = pd.DataFrame(clf.cv_results_)
        # print(df[['param_activation', 'mean_test_score']])
        # print("\n\n")
        # print(clf.best_params_)
        # print(clf.best_score_)

    def tryManyModels(self):
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

        model_params = {
            'neuronal': {
                'model': MLPClassifier(),
                'params': {
                    'activation' :['relu']
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

        g = open('./differentModels.txt', 'a')
        g.write('\n' + str(df))
        g.close()

        stop_time = time.time()
        self.runningTime = round(int(stop_time - start_time)/60, 2)

    def showRunningLogs(self):
        print(f'\nAccuracy: {self.accuracy}')
        print(f'Running time: {self.runningTime}min')

        g = open(self.LOG_FILE_PATH, 'a')
        localtime = time.asctime(time.localtime(time.time()))
        g.write(
            localtime + f'  accuracy: {self.accuracy}, running time: {self.runningTime}\n')
        g.close()

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

    def tryAllFeaturesForSvc(self):
        start_time = time.time()

        self.readData()
        self.readLabels()

        feature_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        best_acc = 0
        best_subset = []
        for L in range(1, len(feature_indexes)+1):
            for subset in itertools.combinations(feature_indexes, L):
                subset = list(subset)
                print(subset)

                self.loadCertainFeaturesAllData(subset)

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

                model = SVC(C=5, gamma='scale', kernel='rbf',
                            decision_function_shape='ovr')
                model.fit(self.train_features, self.train_labels)

                predictions = model.predict(self.validation_features)

                self.accuracy = np.mean(predictions == self.validation_labels)

                predictions = model.predict(self.test_features)

                stop_time = time.time()
                self.runningTime = round(int(stop_time - start_time)/60, 2)

                if(self.accuracy > best_acc):
                    best_acc = self.accuracy
                    best_subset = subset

                g = open('./test_all_feature_combinations_svc.txt', 'a')
                localtime = time.asctime(time.localtime(time.time()))
                g.write(
                    localtime + f'  accuracy: {self.accuracy}, running time: {self.runningTime}, subset: {subset}\n')
                g.close()

        print(best_acc)
        print(best_subset)



pb = Problem()
pb.cleanData()
pb.extractFeaturesAllData()
# pb.svcAlgorithm()
pb.neuralAlgorithm()
# pb.tryManyModels()
# pb.tryAllFeaturesForSvc()
pb.showRunningLogs()
