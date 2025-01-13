import os
import pickle
import pandas as pd
import re


folder1 = "/kyb/agpd/sbruijns/ihmm_essentials/session_data"
folder2 = "/kyb/agpd/sbruijns/ihmm_behav_states/session_data"

files1 = os.listdir(folder1)

pre_bias_sessions = {}
# for every animal, save the number of prebias sessions
for file in files1:
    if not file.endswith("info_dict.p"):
        continue
    with open(os.path.join(folder1, file), "rb") as f:
        data = pickle.load(f)
        
    if file not in os.listdir(folder2):
        print(f"{file} not found in {folder2}")
        continue
    
    with open(os.path.join(folder2, file), "rb") as f:
        data2 = pickle.load(f)
        
    if data['bias_start'] != data2['bias_start']:
        print("uiuiui {}".format(file))
    else:
        print("yay {}".format(file))
    pre_bias_sessions[data['subject']] = data['bias_start']

not_found = []
works = {}
# Compare the contents of the files in the two folders
for file in files1:

    if file.endswith("info_dict.p"):
        continue

    # Read the file from folder1
    with open(os.path.join(folder1, file), "rb") as f:
        data1 = pickle.load(f)
        
    # using a regular expression, we extract the subject name from the file name, which is the part before "_df_[num].p" or before "_fit_info_[num].p"
    matches = re.search(r"(.+?)_(?:df|fit_info|side_info)_(\d+)\.p", file)
    subject = matches.group(1)
    num = matches.group(2)
    # print(file, subject, num)

    if subject not in works:
        works[subject] = []

    # if the file is not in folder2, print a message and skip to the next file
    if file not in os.listdir(folder2):
        print(f"{file} not found in {folder2}")
        not_found.append(subject)
        continue

    # Read the file from folder2
    with open(os.path.join(folder2, file), "rb") as f:
        data2 = pickle.load(f)

    # Compare the contents of the two files, they may be dictionaries or numpy arrays, if they are pandas dataframes, compare only overlapping column names
    if isinstance(data1, dict):
        for key in data1:
            if key not in data2:
                print(f"Key {key} not found in {file}")
                break
            elif not (data1[key] == data2[key]).all():
                print(f"Data for key {key} in {file} does not match")
                break
        else:
            works[subject].append(num)
    elif isinstance(data1, pd.DataFrame):
        choice_word = 'choice' if 'choice' in data2.columns else 'response'
        contL_word = 'contrastL' if 'contrastL' in data2.columns else 'contrastLeft'
        contR_word = 'contrastR' if 'contrastR' in data2.columns else 'contrastRight'
        feedback_word = 'feedback' if 'feedback' in data2.columns else 'feedbackType'
        for n1, n2 in zip(['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'signed_contrast'], [choice_word, contL_word, contR_word, feedback_word, 'signed_contrast']):
            if data1[n1].values.shape != data2[n2].values.shape or not (data1[n1].values == data2[n2].values).all():
                print(f"Data in {file} does not match")
                break
        else:
            works[subject].append(num)
    else:
        # if the file has shape (n, 3), we cut away the last column, making it shape (n, 2)
        if data2.shape[1] == 3:
            data2 = data2[:, :-1]
        # check whether shape of arrays matches, then compare the contents
        if data1.shape != data2.shape or not (data1 == data2).all():
            print(f"Data in {file} does not match")
        else:
            works[subject].append(num)

# for each subject in works, print the first session number not contained in its list of working session numbers
for subject in works:
    works[subject] = sorted(works[subject], key=int)
    if subject not in pre_bias_sessions:
        continue
    for i in range(pre_bias_sessions[subject] + 4):
        if str(i) not in works[subject]:
            # check if this number is lower than the number of prebias sessions of that animal
            if i < pre_bias_sessions[subject]:
                print(f"Subject {subject} has missing session {i + 1} before bias")
                break