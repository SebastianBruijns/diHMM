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

subjects = ['CSHL045', 'CSHL047', 'CSHL049', 'CSHL051', 'CSHL052', 'CSHL053', 'CSHL054', 'CSHL055', 'CSHL058', 'CSHL059', 'CSHL060', 'CSHL_007', 'CSHL_014', 'CSHL_015',
           'CSHL_020', 'CSH_ZAD_001', 'CSH_ZAD_011', 'CSH_ZAD_017', 'CSH_ZAD_019', 'CSH_ZAD_022', 'CSH_ZAD_024', 'CSH_ZAD_025', 'CSH_ZAD_026', 'CSH_ZAD_029', 'DY_008',
           'DY_009', 'DY_010', 'DY_011', 'DY_013', 'DY_014', 'DY_016', 'DY_018', 'DY_020', 'KS014', 'KS015', 'KS016', 'KS017', 'KS019', 'KS021', 'KS022', 'KS023',
           'KS042', 'KS043', 'KS044', 'KS045', 'KS046', 'KS051', 'KS052', 'KS055', 'KS084', 'KS086', 'KS091', 'KS094', 'KS096', 'MFD_05', 'MFD_06', 'MFD_07', 'MFD_08',
           'MFD_09', 'NR_0017', 'NR_0019', 'NR_0020', 'NR_0021', 'NR_0024', 'NR_0027', 'NR_0028', 'NR_0029', 'NR_0031', 'NYU-06', 'NYU-11', 'NYU-12', 'NYU-21', 'NYU-27',
           'NYU-30', 'NYU-37', 'NYU-39', 'NYU-40', 'NYU-45', 'NYU-46', 'NYU-47', 'NYU-48', 'NYU-65', 'PL015', 'PL016', 'PL017', 'PL024', 'PL030', 'PL031', 'PL033',
           'PL034', 'PL035', 'PL037', 'PL050', 'SWC_021', 'SWC_022', 'SWC_023', 'SWC_038', 'SWC_039', 'SWC_042', 'SWC_043', 'SWC_052', 'SWC_053', 'SWC_054', 'SWC_058',
           'SWC_060', 'SWC_061', 'SWC_065', 'SWC_066', 'UCLA005', 'UCLA006', 'UCLA011', 'UCLA012', 'UCLA014', 'UCLA015', 'UCLA017', 'UCLA030', 'UCLA033', 'UCLA034',
           'UCLA035', 'UCLA036', 'UCLA037', 'UCLA044', 'UCLA048', 'UCLA049', 'UCLA052', 'ZFM-01576', 'ZFM-01577', 'ZFM-01592', 'ZFM-01935', 'ZFM-01936', 'ZFM-01937',
           'ZFM-02368', 'ZFM-02369', 'ZFM-02370', 'ZFM-02372', 'ZFM-02373', 'ZFM-04308', 'ZFM-05236', 'ZM_1897', 'ZM_1898', 'ZM_2240', 'ZM_2241', 'ZM_2245', 'ZM_3003',
           'ibl_witten_13', 'ibl_witten_14', 'ibl_witten_16', 'ibl_witten_17', 'ibl_witten_18', 'ibl_witten_19', 'ibl_witten_20', 'ibl_witten_25', 'ibl_witten_26',
           'ibl_witten_27', 'ibl_witten_29', 'ibl_witten_32']

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

compute_wsls = False
add_on = "" if not compute_wsls else "_wsls"

compute_exp = False
add_on += "" if compute_exp else "_no_exp"

num_sess = []
# if this level of exponential decay has no folder, make it
if not os.path.exists("./summarised_sessions/{}{}/".format(str(exponential_decay).replace(".", "_"), add_on)):
    os.makedirs("./summarised_sessions/{}{}/".format(str(exponential_decay).replace(".", "_"), add_on))

for subject in subjects:
    all_sessions = []
    info_dict = pickle.load(open("./session_data/{}_info_dict.p".format(subject), "rb"))

    # Determine session numbers
    if fit_type == 'prebias':
        till_session = info_dict['bias_start']
    elif fit_type == 'bias' or fit_type == 'zoe_style' or fit_type == 'all':
        till_session = info_dict['ephys_start']

    from_session = info_dict['bias_start'] if fit_type in ['bias', 'zoe_style'] else 0
    num_sess.append(till_session - from_session)
    print(subject, till_session - from_session)

    subject_timeouts = {}

    for j in range(from_session, till_session):
        # load the _fit_info file
        with open("./session_data/{}_fit_info_{}.p".format(subject, j), "rb") as f:
            data = pickle.load(f)

        with open("./session_data/{}_side_info_{}.p".format(subject, j), "rb") as f:
            side_info = pickle.load(f)

        assert data.shape[0] != 0, "Data is empty for subject {} and session {}".format(subject, j)

        data[:, 0] = cont_mapping(data[:, 0])
        mask = data[:, 1] != 1  # throw out timeout trials
        if fit_type == 'zoe_style':
            mask[90:] = False

        mega_data = np.empty((np.sum(mask), n_regressors + 2 + compute_wsls))  # session number, number of regressors, and choice

        mega_data[:, 0] = j  # session number
        mega_data[:, 1] = np.maximum(data[mask, 0], 0)  # rightwards contrasts are positive numbers, otherwise put 0
        mega_data[:, 2] = np.abs(np.minimum(data[mask, 0], 0))  # leftwards contrasts are negative numbers, otherwise put 0

        # exponentially filtered previous answers
        new_prev_ans = data[:, 1].copy()
        new_prev_ans -= 1  # go from 0 2 encoding to -1 1 encoding
        new_prev_ans = smooth_exp_array(new_prev_ans, exponential_decay)

        # note down timeout responses for predictive checks
        subject_timeouts[j] = {}
        if np.sum(~mask) > 0:
            print(np.sum(~mask), j)
            actual_trial_counter = 0
            skip_counter = 0
            for m in mask:
                if m:
                    if skip_counter > 0:
                        subject_timeouts[j][actual_trial_counter] = skip_counter
                        skip_counter = 0
                    actual_trial_counter += 1
                else:
                    skip_counter += 1
            
        mega_data[:, 3] = new_prev_ans[mask]

        # bias
        mega_data[:, 4] = 1

        # wsls
        if compute_wsls:
            answers = data[:, 1].copy() - 1
            rewards = side_info[:, 1]
            wsls = np.zeros(data.shape[0])
            wsls[1:] = answers[:-1]
            wsls[1:][rewards[:-1] == 0] *= -1
            mega_data[:, 5] = wsls[mask]

        # animal choice, transformed into 0 1 encoding
        mega_data[:, -1] = data[mask, 1] / 2

        all_sessions.append(mega_data)

    # turn the list of sessions into a dataframe with correct column names
    all_sessions = np.vstack(all_sessions)
    cols = ['session', 'right_contrast', 'left_contrast', 'prev_ans', 'bias'] + ['wsls'] * compute_wsls + ['choice']
    df = pd.DataFrame(all_sessions, columns=cols)

    # if not compute_exp, remove the exponential decay column
    if not compute_exp:
        df = df.drop(columns=['prev_ans'])

    # save the dataframe as csv
    df.to_csv("./summarised_sessions/{}{}/{}_{}_fit_info.csv".format(str(exponential_decay).replace('.', '_'), add_on, subject, fit_type), index=False)
    pickle.dump(subject_timeouts, open(f"./session_data/subject_timeout_{subject}", 'wb'))
