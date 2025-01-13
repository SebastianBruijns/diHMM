"""
    Read in all the individual behavioural data files, and turn them into the desired regressors for the model fit.
"""
import numpy as np
import pickle
import json
import os
import pandas as pd

# following Nick Roys contrasts: following tanh transformation of the contrasts x has a
# free parameter p which we set as p= 5 throughout the paper: xp = tanh (px)/ tanh (p).
contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}
cont_mapping = np.vectorize(num_to_contrast.get)

lenca_data = pd.read_csv("./lenca_data/Ibl_processed.csv")
subjects = lenca_data['subject'].unique()
subjects = ['NYU-07', 'NYU-12', 'CSHL_003', 'NYU-13', 'CSHL_015', 'NYU-04', 'ibl_witten_13', 'ibl_witten_06', 'ibl_witten_12', 'CSHL_014', 'CSHL_010', 'ibl_witten_16',
            'CSHL_005', 'CSHL_004', 'NYU-02', 'ibl_witten_15', 'ibl_witten_14', 'CSHL_012', 'IBL-T1', 'IBL-T3', 'IBL-T2', 'NYU-09', 'CSHL_008']

def smooth_exp_array(array, decay):
    """
    Smooth an array of 0s and 1s with an exponential filter, infinitely.
    
    :param array: List or numpy array of 0s and 1s
    :param window_width: Integer, the width of the rectangular filter
    :return: Numpy array of the smoothed values
    """
    
    # Convert input to numpy array for easier manipulation
    array = np.array(array)
    
    # Create an array to hold the smoothed values
    smoothed_array = np.zeros_like(array, dtype=float)
    
    # bit awkward, but we need to normalise the filter to area 1 as we did previously
    normalisation = np.exp(- decay * np.arange(5000)).sum()
    decay_const = np.exp(- decay) # we the filter to area 1, so the features are on an even field
    # Apply the rectangular filter (moving average)
    for i in range(1, len(array)):
        # Store the smoothed value
        # print(array[i-1], normalisation, decay_const, smoothed_array[i - 1])
        smoothed_array[i] = array[i-1] / normalisation + decay_const * smoothed_array[i - 1]
    
    return smoothed_array

fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][0]
n_regressors = 4
exponential_decay = 0.25

# if this level of exponential decay has no folder, make it
if not os.path.exists("./summarised_sessions_lenca/{}/".format(str(exponential_decay).replace(".", "_"))):
    os.makedirs("./summarised_sessions_lenca/{}/".format(str(exponential_decay).replace(".", "_")))

for subject in subjects:
    all_sessions = []
    print(subject)
    info_dict = pickle.load(open("./session_data_lenca_for_me/{}_info_dict.p".format(subject), "rb"))

    # Determine session numbers
    if fit_type == 'prebias':
        till_session = info_dict['bias_start']
    elif fit_type == 'bias' or fit_type == 'zoe_style' or fit_type == 'all':
        till_session = info_dict['ephys_start']

    from_session = info_dict['bias_start'] if fit_type in ['bias', 'zoe_style'] else 0

    for j in range(from_session, till_session):
        # load the _fit_info file
        with open("./session_data_lenca_for_me/{}_fit_info_{}.p".format(subject, j), "rb") as f:
            data = pickle.load(f)
        
        assert data.shape[0] != 0, "Data is empty for subject {} and session {}".format(subject, j)

        data[:, 0] = cont_mapping(data[:, 0])
        mask = data[:, 1] != 1  # throw out timeout trials
        if fit_type == 'zoe_style':
            mask[90:] = False

        mega_data = np.empty((np.sum(mask), n_regressors + 2))  # session number, number of regressors, and choice

        mega_data[:, 0] = j  # session number
        mega_data[:, 1] = np.maximum(data[mask, 0], 0)  # rightwards contrasts are positive numbers, otherwise put 0
        mega_data[:, 2] = np.abs(np.minimum(data[mask, 0], 0))  # leftwards contrasts are negative numbers, otherwise put 0

        # exponentially filtered previous answers
        new_prev_ans = data[:, 1].copy()
        new_prev_ans -= 1  # go from 0 2 encoding to -1 1 encoding
        new_prev_ans = smooth_exp_array(new_prev_ans, exponential_decay)
        mega_data[:, 3] = new_prev_ans[mask]

        # bias
        mega_data[:, 4] = 1

        # animal choice, transformed into 0 1 encoding
        mega_data[:, -1] = data[mask, 1] / 2

        all_sessions.append(mega_data)

    # turn the list of sessions into a dataframe with correct column names
    all_sessions = np.vstack(all_sessions)
    df = pd.DataFrame(all_sessions, columns=['session', 'right_contrast', 'left_contrast', 'prev_ans', 'bias', 'choice'])

    # save the dataframe as csv
    df.to_csv("./summarised_sessions_lenca/{}/{}_{}_fit_info.csv".format(str(exponential_decay).replace('.', '_'), subject, fit_type), index=False)
