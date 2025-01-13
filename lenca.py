import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import re

# Regular expression pattern to capture everything before "_data_and_indices_CV_5_folds"
pattern = r"(.*)_data_and_indices_CV_5_folds"
names = []
equal_mice = []
lol = []

contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}
cont_mapping = np.vectorize(num_to_contrast.get)

for file in os.listdir("./lenca_data/"):
    if not file.endswith(".npz"):
        continue

    # Search for the pattern in the example string
    mouse_name = re.search(pattern, file).group(1)
    names.append(mouse_name)

    if mouse_name in ['CSHL_002']:
        # some mice don't match at all, some contrasts seem flipped
        continue

    if not os.path.isfile("/kyb/agpd/sbruijns/ihmm_essentials/session_data_lenca_for_me/" + "{}_fit_info_{}.p".format(mouse_name, 0)):
        print("skipped " + mouse_name)
        continue

    data = np.load("./lenca_data/" + file)

    trials = []
    all_responses = []
    all_contrasts = []
    for i in range(100):
        try:
            my_data = pickle.load(open("/kyb/agpd/sbruijns/ihmm_essentials/session_data_lenca_for_me/" + "{}_fit_info_{}.p".format(mouse_name, i), "rb"))
            trials.append((my_data[:, 1] != 1).sum())
            all_responses.append(my_data[:, 1])
            all_contrasts.append(my_data[:, 0])
        except:
            pass

    info_dict = pickle.load(open("./session_data_lenca_for_me/{}_info_dict.p".format(mouse_name), "rb"))

    if 'bias_start' not in info_dict:
        lol.append(mouse_name)
        continue

    n_sess = min(data['sessInd'].shape[0] - 1, info_dict['bias_start'])

    print(mouse_name, (data['sessInd'][1:n_sess] == np.cumsum(trials)[:n_sess-1]).mean())

    if (data['sessInd'][1:n_sess] == np.cumsum(trials)[:n_sess-1]).mean() == 1.:
        session_counter = 0
        for i in range(n_sess):
            valid_responses = all_responses[i][all_responses[i] != 1]
            assert np.array_equal(valid_responses // 2, (data['y'][session_counter:session_counter + valid_responses.shape[0]] + 1) % 2)
            lenca_conts = data['x2'][session_counter:session_counter + valid_responses.shape[0], [1, 2]]
            assert np.allclose(cont_mapping(all_contrasts[i][all_responses[i] != 1]), lenca_conts[:, 0] - lenca_conts[:, 1], atol=0.001)  # needed to increase tolerance a bit, I think we round differently
            session_counter += valid_responses.shape[0]
        print("responses and contrasts are the same")
        equal_mice.append(mouse_name)

    # if mouse_name == "NYU-12":
    #     quit()

print(names)
print(equal_mice)