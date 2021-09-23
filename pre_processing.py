"""
I, Omar LInes, have read and understood the School's Academic Integrity Policy,
as well as guidance relating to this module, and confirm that this submission
complies with the policy.

The content of this file is my own original work, with any significant material
copied or adapted from other sources clearly indicated and attributed.
"""
import time

import numpy as np


import os
import librosa
import librosa.feature as lib_f
import main
np.random.seed(123)


def run_pre_processing():

    raw_X, Y = raw_extract()
    encoding_labels(Y)
    extract_features(raw_X)


def raw_extract():

    features = []
    classLabels = []

    for i in os.listdir(os.getcwd()+"\\genres\\"):
        for ii in os.listdir(os.getcwd()+"\\genres\\"+i):
            features.append(ii)
            classLabels.append(i)

    # Create feature vector of audio files and class vector of labels.
    X = np.asarray(features)
    Y = np.asarray(classLabels)

    return X, Y


# Function converts class labels to a vector of categorical int values
# e.g. rock = 0, jazz = 1, hip-hop = 2 etc.
def encoding_labels(Y):

    labels = np.unique(Y)
    encodeDict = dict(zip(labels, range(len(labels))))
    Y = np.array([encodeDict[label] for label in Y])

    # save encoded labels.
    np.save("Encoded Labels",  Y)

    # Better wat to encode labels but wanted to see if i could do it manually
    # classes, Y_encoded = np.unique(Y, return_inverse=True)


# extracts features from each audio files
def extract_features(X):

    set_range = np.array((0, 100))
    threshold = 99

    to_append = []
    feature_headings = ["chroma_stft",
                        "chroma_cqt",
                        "chroma_cens",
                        "mel_freq",
                        "root_mean_squared",
                        "bandwidth",
                        "centroid",
                        "nth_order_polynomial",
                        "zcr",
                        "rhythme_feature"]

    # feature extraction execution time:  ~28.385 minutes
    start_time = time.time()
    for i in os.listdir(os.getcwd()+"\\genres\\"):
        for ii in range(set_range[0], set_range[1]):

            new_path = os.getcwd()+"\\genres\\"+i+"\\"+X[ii]

            # sample rate = 22050Hz
            X, sample_rate = librosa.load(new_path, mono=True, duration=30)

            # print statement to track progress.
            print("Extracting features from {}".format(X[ii]))

            # featrues extracted. More info here:
            # https://librosa.org/doc/latest/feature.html
            chroma_stft = lib_f.chroma_stft(X, sample_rate)
            chroma_cqt = lib_f.chroma_cqt(X, sample_rate)
            chroma_cens = lib_f.chroma_cens(X, sample_rate)
            mel_freq = lib_f.melspectrogram(X, sample_rate)
            root_mean_squared = lib_f.rms(X, sample_rate)
            bandwidth = lib_f.spectral_bandwidth(X, sample_rate)
            centroid = lib_f.spectral_centroid(X, sample_rate)
            nth_order_polynomial = lib_f.poly_features(X, sample_rate)
            zcr = lib_f.zero_crossing_rate(X)
            rhythme_feature = lib_f.tempogram(X, sample_rate)
            mfcc = lib_f.feature.mfcc(X, n_mfcc=10)

            to_append.extend((np.mean(chroma_stft),
                              np.mean(chroma_cqt),
                              np.mean(chroma_cens),
                              np.mean(mel_freq),
                              np.mean(root_mean_squared),
                              np.mean(bandwidth),
                              np.mean(centroid),
                              np.mean(nth_order_polynomial),
                              np.mean(zcr),
                              np.mean(rhythme_feature)))

            for e in mfcc:
                to_append.append(np.mean(e))

            # this is a bad way to iterate through each fold but it works.
            # Logic is correct, executuion is poor.
            # P.S look for a better solution if there is time at the end.
            if ii == threshold:
                set_range += 100
                threshold += 100

    feature_values = np.asarray(to_append).reshape(X.shape[0],
                                                   len(feature_headings)+10)

    np.save("Extracted Features", feature_values)
    print("Feature Extraction runtime: --- %s minutes ---" %
          ((time.time() - start_time) / 60))
