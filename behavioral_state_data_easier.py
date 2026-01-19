"""
    Script for downloading the data of the paper "Dissecting the Complexities of Learning With Infinite Hidden Markov Models"
    Download this using the IBL environment: https://github.com/int-brain-lab/iblenv
"""
from one.api import ONE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import json
import re
import warnings
import os

warnings.simplefilter(action='ignore')

# the password can be found at https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html
one = ONE()

regexp = re.compile(r'Subjects/\w*/((\w|-)+)/_ibl')
datasets = one.alyx.rest('datasets', 'list', tag='2023_Q4_Bruijns_et_al')

# extract subject names
subjects = [regexp.search(ds['file_records'][0]['relative_path']).group(1) for ds in datasets]
# reduce to list of unique names
subjects = list(set(subjects))
data_folder = 'session_data'
contrast_to_num = {-1.: 0, -0.5: 1, -0.25: 2, -0.125: 3, -0.0625: 4, 0: 5, 0.0625: 6, 0.125: 7, 0.25: 8, 0.5: 9, 1.: 10}

os.makedirs(data_folder, exist_ok=True)
os.makedirs('./figures/behavior/', exist_ok=True)

for subject in subjects:
    trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

    # Load training status and join to trials table
    training = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')

    trials = (trials
              .set_index('session')
              .join(training.set_index('session'))
              .sort_values(by='session_start_time', kind='stable'))

    # use np.unique to find unique start_times and eids, each session only has one. Then sort those.
    start_times, indices = np.unique(trials.session_start_time, return_index=True)
    start_times = [trials.session_start_time[index] for index in sorted(indices)]
    eids, indices = np.unique(trials.index, return_index=True)
    eids = [trials.index[index] for index in sorted(indices)]

    performance = np.zeros(len(eids))
    easy_per = np.zeros(len(eids))
    hard_per = np.zeros(len(eids))

    info_dict = {'subject': subject, 'dates': [st.to_pydatetime() for st in start_times], 'eids': eids, 'date_and_session_num': {}}
    contrast_set = {0, 1, 9, 10}  # starting contrasts
    to_introduce = [2, 3, 4, 5]  # these contrasts need to be introduced, keep track when that happens

    for i, start_time in enumerate(start_times):

        df = trials[trials.session_start_time == start_time]
        df.loc[:, 'contrastRight'] = df.loc[:, 'contrastRight'].fillna(0)
        df.loc[:, 'contrastLeft'] = df.loc[:, 'contrastLeft'].fillna(0)
        df.loc[:, 'feedbackType'] = df.loc[:, 'feedbackType'].replace(-1, 0)
        df.loc[:, 'signed_contrast'] = df.loc[:, 'contrastRight'] - df.loc[:, 'contrastLeft']
        df.loc[:, 'signed_contrast'] = df.loc[:, 'signed_contrast'].map(contrast_to_num)
        df.loc[:, 'choice'] = df.loc[:, 'choice'] + 1

        if any([df[x].isnull().any() for x in ['signed_contrast', 'choice', 'feedbackType', 'probabilityLeft']]):
            quit()

        # check whether new contrasts got introduced
        current_contrasts = set(df['signed_contrast'])
        diff = current_contrasts.difference(contrast_set)
        for c in to_introduce:
            if c in diff:
                info_dict[c] = i
        contrast_set.update(diff)

        # document performance for plotting
        performance[i] = np.mean(df['feedbackType'])
        easy_per[i] = np.mean(df['feedbackType'][np.logical_or(df['signed_contrast'] == 0, df['signed_contrast'] == 10)])
        hard_per[i] = np.mean(df['feedbackType'][df['signed_contrast'] == 5])

        print(df.task_protocol.iloc[0])

        if 'bias_start' not in info_dict and (df.task_protocol.iloc[0].startswith('_iblrig_tasks_biasedChoiceWorld') or df.task_protocol.iloc[0].startswith('biasedChoiceWorld')):
            info_dict['bias_start'] = i
            print('start')

        if 'ephys_start' not in info_dict and (df.task_protocol.iloc[0].startswith('_iblrig_tasks_ephysChoiceWorld') or df.task_protocol.iloc[0].startswith('ephysChoiceWorld')):
            info_dict['ephys_start'] = i

        pickle.dump(df, open("./{}/{}_df_{}.p".format(data_folder, subject, i), "wb"))
        
        info_dict['date_and_session_num'][i] = str(start_time)
        info_dict['date_and_session_num'][str(start_time)] = i

        side_info = np.zeros((len(df), 2))
        side_info[:, 0] = df['probabilityLeft']
        side_info[:, 1] = df['feedbackType']
        pickle.dump(side_info, open("./{}/{}_side_info_{}.p".format(data_folder, subject, i), "wb"))

        fit_info = np.zeros((len(df), 2))
        fit_info[:, 0] = df['signed_contrast']
        fit_info[:, 1] = df['choice']
        pickle.dump(fit_info, open("./{}/{}_fit_info_{}.p".format(data_folder, subject, i), "wb"))

    info_dict['n_sessions'] = i
    pickle.dump(info_dict, open("./{}/{}_info_dict.p".format(data_folder, subject), "wb"))

    # optional plotting of the evolution of performance across sessions
    plt.figure(figsize=(11, 8))
    plt.plot(performance, label='Overall')
    plt.plot(easy_per, label='100% contrasts')
    plt.plot(hard_per, label='0% contrasts')
    plt.axvline(info_dict['bias_start'] - 0.5)

    for c in to_introduce:
        plt.axvline(info_dict[c], ymax=0.85, c='grey')
    plt.annotate('Pre-bias', (info_dict['bias_start'] / 2, 1.), size=20, ha='center')
    plt.annotate('Bias', (info_dict['bias_start'] + (i - info_dict['bias_start']) / 2, 1.), size=20, ha='center')
    plt.title(subject, size=22)
    plt.ylabel('Performance', size=22)
    plt.xlabel('Session', size=22)
    plt.xticks(size=16)
    plt.xticks(size=16)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    sns.despine()
    plt.tight_layout()
    plt.savefig('./figures/behavior/all_of_trainig_{}'.format(subject))
    plt.close()
