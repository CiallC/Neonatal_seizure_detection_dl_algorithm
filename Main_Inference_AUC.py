# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 11:25:11 2023
AUC calcs for checking
@author: Aengus.Daly

"""

import os
import os.path
import scipy.io as sio
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # To be used for GPU use
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from keras.models import Model, load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import np_utils
import time
from Utils.ConvNet import res_net
from Utils.score_tool import calc_roc
from Utils.Inference_probs import mean_maf_probability

start_time = time.time()
weights_path = './Benchmark_weights/'  # Folder with the model weights
eeg_file_path = 'EEG_files/'  # Folder with EEG signal data
results_path = './Results/' # Folder for storing results output
model_path = 'Utils/ConvNet_model.keras'  # Path for keras model
file_list = ["eeg1_SIGNAL.mat", "eeg4_SIGNAL.mat"] #
epoch_length = 16  # Length of epoch/window input of EEG signal in seconds
epoch_shift = 1 # Epoch/window shift in seconds
input_sampling_rate = 32  # 32 Hz is the input signal sampling rate
sample_rate = 1 # Sampling rate (of 32 Hz) used to make data epoch/window slices in TSG, usually 1
runs = 3  # No. of model weights used; weights were generated via different random initializations during training runs.
window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter
model_weights_files = [] # initialize
weights_list = {"1": ['best_weights_run0_hski_trained.hdf5'],
                    "2": ['best_weights_run0_hski_trained.hdf5','best_weights_run1_hski_trained.hdf5'],
                    "3": ['best_weights_run0_hski_trained.hdf5','best_weights_run1_hski_trained.hdf5','best_weights_run2_hski_trained.hdf5']}
                # Dictionary -  runs: List of model weights' file names

def getdata(baby,eeg_file_path):
    '''
    Function to process the EEG signal data
    :param baby: file name of the EEG signal data
    :param eeg_file_path: file_path for the above file name
    :return: data after reshaping for TSG, no. of eeg channels
    '''

    trainX = []
    trainY = []

    X = sio.loadmat(eeg_file_path+str(baby))['EEG'] # X is 32 Hz EEG signal and 18 channels
    no_eeg_channels = np.shape(X)[1]
    trainX.extend(X.reshape(len(X),no_eeg_channels,1))

    try:
        Y = sio.loadmat(eeg_file_path + str('annotations_2017.mat'))['annotat_new'][0][
            int(baby[3]) - 1]  # Y is the label, 1 per second, choose baby with index Baby-1
        Y = np.sum(Y, axis=0)  # For consensus anns
        Y = np.where(Y == 3, 1, 0)  # For consensus anns
    except:
        Y = []
    if int(np.shape(X)[0] / input_sampling_rate) != len(Y):
        print('Anns length different to EEG length', len(X), int(np.shape(X)[0] / input_sampling_rate))

    trainY.extend(Y.repeat(input_sampling_rate))

    return trainX, trainY, no_eeg_channels

def movingaverage(data, epoch_length):
    '''
    Moving average filter used on outputted probabilites
    :param data: the vector which the MAF will be applied to
    :param window: the size of the moving average window
    :return: data after the MAF has been applied
    '''
    data = data
    window = 69 - epoch_length
    window = np.ones(int(window)) / float(window)
    return np.convolve(data, window, "same")


def inference():
    ''' Primary function, run below via __main__'''

    probs_full = []

    for baby in (file_list):

        print('EEG file started inference....', baby)
        print("--- %.0f seconds ---" % (time.time() - start_time))

        testX, testY, no_eeg_channels = getdata(baby, eeg_file_path)
        if os.path.isfile(model_path) == True:  # compile and save model file if it does not exist
            model = res_net(no_eeg_channels,input_length=epoch_length*input_sampling_rate)
            model.save(model_path)
        model = load_model(model_path)
        print(model.summary())

        # Generate data slices of (length= epoch_length*input_sampling_rate), shifted by (stride = input_sampling rate*epoch_shift), and sampling rate = sample rate
        # testY = np.ones((len(testX))) # this is needed for TimeSeriesGenerator (TSG)
        data_gen = TimeseriesGenerator(np.asarray(testX), np_utils.to_categorical(np.asarray(testY)),
                                       length=epoch_length*input_sampling_rate, sampling_rate=sample_rate, stride=input_sampling_rate*epoch_shift, batch_size=300, shuffle=False)
        probs_full = []
        downsampled_y_full = []
        for weights_str in weights_list[str(runs)]:
            model.load_weights(weights_path + weights_str)

            probs = model.predict(data_gen)[:, 1]
            probs = movingaverage(probs, epoch_length)  # Applying moving average filter to probs
            probs_full.append(probs) #appending probs so that they can be averaged

        probs_full = np.asarray(probs_full)
        probs_full = np.mean(probs_full, 0)
        probs_full = probs_full
        # probs_full = np.append(probs_full, probs) # to be used for concatenating probs

        downsampled_y = testY[::input_sampling_rate][:-epoch_length]
        downsampled_y_full = np.append(downsampled_y_full, downsampled_y)

        AUC = calc_roc(probs_full, downsampled_y_full, epoch_length=epoch_length)  # Removed MAF
        print('AUC %f, AUC90 %f' % (AUC))
        print('For no. of model runs...', runs)
        results_file_name = results_path + 'probs_'+ baby[:-4] + '.npy'
        np.save(results_file_name, probs_full)
        print('Probabilities created in folder/file....',results_file_name)
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("--- %.0f seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    # LOGGER.info("Script started...")

    # run_model = True
    # save_results = False
    # check_options(run_models, save_results)

    inference()