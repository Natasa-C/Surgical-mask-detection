
# Surgical mask detection
#### Discriminate between utterances with and without surgical mask

## Data Description
#### Task
Participants have to train a model for surgical mask detection. This is a binary classification task in which an utterance (audio file) must be labeled as without mask (label 0) or with mask (label 1).

The training data is composed of 8000 audio files. The validation set is composed of 1000 audio files. The test is composed of 3000 audio files.

### File Descriptions
- train.txt - the training metadata file containing the audio file names and the corresponding labels (one example per row)
- validation.txt - the validation metadata file containing the audio file names and the corresponding labels (one example per row)
- test.txt - the test metadata file containing the audio file names (one sample per row)
- sample_submission.txt - a sample submission file in the correct format

### Data Format
Metadata Files
The metadata files are provided in the following format based on comma separated values:

- 1.wav,0
- 2.wav,1

Each line represents an example where:
- The first column shows the audio file name of the example.
- The second column is the label associated to the example.

### Audio Files
The audio files are provided in .wav format.

# Research and code implementation

## [1] Initial preprocessing: cleaning the data

We perceive sound in the frequency domain. The cochlea in our ear actually performs a biological Fourier transform by converting sound waves into neural signals of frequency amplitudes. It can be useful to also process digital audio signals in the frequency domain. For example tuning the lows, mids, and highs of an audio signal could be done by performing a Fourier transform on the time domain samples, scaling the various frequencies as desired, and then converting back to an audio signal with an inverse Fourier transform.

[ **envelope**] create an envelope function used to create a mask with True/False values which will be used to reduce the empty/under the threshold portions of data
    - convert the numpy array into a series and transform each value in the series into it's absolute value 
    - create a rolling window over the signal with pandas which provides rolling window calculations and get the mean of the window (window = window size is going to be a tenth of a second which translates to a tenth of the collection rate samples (we have 44100 samples/second, so in a tenth o a second, we go over a tenth of them), min_periods = the minimum number of values that we need in our window to create a calculation, center = center the window)

[**clean the data**] the function is going to create True/False masks (envelopes) to keep only the relevant parts of the signal from the .wav files using a ```threshold = 0.0005``` and write the created clean files into the specified folder. Once the data has been cleaned, we changed the paths to the data to point to the clean ones.

Resources:
- [envelope function] https://www.youtube.com/watch?v=mUXkj1BKYk0&t=289s
- [series/rolling window] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html


## [2] .WAV file spectrogram and feature extraction
A spectrogram is a visual way of representing the signal strength, or “loudness”, of a signal over time at various frequencies present in a particular waveform. Every audio signal consists of many features from which we must extract the characteristics that are relevant to the problem we are trying to solve. The spectral features (frequency-based features) are obtained by converting the time-based signal into the frequency domain using the Fourier Transform.

The functions used to extract the features come, mainly, from ```librosa.feature``` library.

The features are extracted in separate directories for the raw data and for the clean data and stored in .csv files. The general features are stored separately from the mfcc values.

Resources:
- [librosa.feature] https://librosa.github.io/librosa/feature.html
-  [definitions and implementation examples] https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

### [2.1] Spectral Centroid
The spectral centroid is commonly associated with the measure of the brightness of a
sound. This measure is obtained by evaluating the “center of gravity” using the Fourier
transform’s frequency and magnitude information. 

In practice, centroid finds this frequency for a given frame, and then finds the nearest
spectral bin for that frequency. The centroid is usually a lot higher than one might
intuitively expect, because there is so much more energy above (than below) the
fundamental which contributes to the average.

Intuitive: It is center of mass of the spectrum. Since spectrum gives the indication of how the signal's mass (amplitude) is distributed among the frequencies, its center of mass indicates the average amount of amplitude. From speech perspective, it is the average loudness. From the image perspective, it is the average brightness. The mathematical equation used is, as you must be knowing or have guessed by now, weighted average. 'Weighted' because, the frequency components may be for instance non-uniformly separated (depending upon the transformation used) or due to application of filters sometimes it makes more sense to use the frequency information also into average instead of giving equal importance.

Resources:
- [Spectral Centroid] https://ccrma.stanford.edu/~unjung/AIR/areaExam.pdf
- [Spectral Centroid] https://www.quora.com/In-an-intuitive-explanation-what-is-spectral-centroid

### [2.2] Spectral Rolloff
It is a measure of the shape of the signal. It represents the frequency at which high frequencies decline to 0. To obtain it, we have to calculate the fraction of bins in the power spectrum where 85% (the default in ```librosa.feature.spectral_rolloff```) of its power is at lower frequencies. Intuitively, the roll-off frequency is defined as the frequency under which some percentage (cutoff) of the total energy of the spectrum is contained. The roll-off frequency can be used to distinguish between harmonic (below roll-off) and noisy sounds (above roll-off)

Resources:
- [roll-off frequency] https://essentia.upf.edu/reference/streaming_RollOff.html

### [2.3] Spectral Bandwidth
The spectral bandwidth is defined as the width of the band of light at one-half the peak maximum (or full width at half maximum [FWHM]) and is represented by the two vertical red lines and λSB on the wavelength axis.

Resources:
- https://www.analiticaweb.com.br/newsletter/02/AN51721_UV.pdf

### [2.4] Zero-Crossing Rate
By looking at different speech and audio waveforms, we can see that depending on the content, they vary a lot in their **smoothness**. For example, voiced speech sounds are more smooth than unvoiced ones. Smoothness is thus a informative characteristic of the signal.

A very simple way for measuring the smoothness of a signal is to calculate the number of zero-crossing within a segment of that signal. A voice signal oscillates slowly — for example, a 100 Hz signal will cross zero 100 per second — whereas an unvoiced fricative can have 3000 zero crossings per second.

Resources:
- [Zero-Crossing Rate] https://wiki.aalto.fi/display/ITSP/Zero-crossing+rate

### [2.5] Mel-Frequency Cepstral Coefficients(MFCCs)
The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.

Resources:
- [explanation] https://www.youtube.com/watch?v=m3XbqfIij_Y
- [implementation example] https://www.youtube.com/watch?v=Oa_d-zaUti8
- [librosa.feature.mfcc] https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
- [tutorial] http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
- [explanation with graphics] https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f

### [2.6] Chroma feature
A chroma feature or vector is typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, …, B}, is present in the signal. In short, It provides a robust way to describe a similarity measure between music pieces.

### [2.7] FFT - frequency and magnitude
A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.

Resources:
- [fft] https://www.karlsims.com/fft.html
- [fft] https://en.wikipedia.org/wiki/Fast_Fourier_transform
- [numpy.fft] https://numpy.org/doc/stable/reference/routines.fft.html
- [explanations and examples] https://www.mathworks.com/help/signal/examples/practical-introduction-to-frequency-domain-analysis.html

### [2.8] Spectral Contrast
Spectral contrast considers the spectral peak, the spectral valley, and their difference in each frequency subband. For more information.

Resources:
- https://musicinformationretrieval.com/spectral_features.html


## [4] Hyperparameter Tuning

I used ```sklearn.model_selection.GridSearchCV``` to try different parameters for ```sklearn.svm.SVC``` and ```sklearn.neural_network.MLPClassifier``` models.

Resources:
- https://github.com/codebasics/py/blob/master/ML/15_gridsearch/Exercise/15_grid_search_cv_exercise.ipynb
- https://github.com/codebasics/py/blob/master/ML/15_gridsearch/15_grid_search.ipynb
- https://www.youtube.com/watch?v=HdlDYng8g9s
- https://www.youtube.com/watch?v=pooXM9mM7FU
- https://www.kaggle.com/funxexcel/p2-logistic-regression-hyperparameter-tuning

## [5] Standardization/Normalization
I used ```sklearn.preprocessing.scale```, ```sklearn.preprocessing.StandardScaler``` and ```sklearn.preprocessing.normalize``` for scaling and normalizing data.

Resources:
- [standardization/normalization] https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/?fbclid=IwAR31clqIFgUfDgvh4GoU4TY-Qgse1qOuDdQp6wVu8qzr2BnxBfkZFOX9hYU

## [6] Ploting

I used ```skplt.metrics.plot_precision_recall_curve``` and ```skplt.metrics.plot_confusion_matrix``` for plotting in order to analyze parameters.

Resources:
 - https://github.com/reiinakano/scikit-plot/issues/87


## [other related] Resources:
- https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html?fbclid=IwAR1w6-IcQ3yvuH2frW3vEDl7CqeC4yY6KOnrrZKSMz2b_MHO6qadmj-PSKg
- https://dev.to/zenulabidin/python-audio-processing-at-lightspeed-part-1-zignal-5658

https://musicinformationretrieval.com/spectral_features.html



