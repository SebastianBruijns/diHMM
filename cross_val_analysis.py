import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

folder = "./lenca_cross_val_infos/"

perfs = []
perfs_2 = []
trial_wise_nll = {}

twelvek_samples = []
sixteenk_samples = []

for file in os.listdir(folder):
    pattern = r"(.*)_(\d)_cvll_(.*)_prebias_(0\.03|0\.04)_.*.json"

    matches = re.search(pattern, file)
    mouse_name = matches.group(1)
    fold = matches.group(2)
    ll = matches.group(3)

    print(mouse_name, fold, ll)

    infos = json.load(open(folder + file, 'r'))

    if matches.group(4) != '0.03':
        perfs_2.append(np.exp(np.mean(infos['cross_val_preds'][-1000:])))
        continue

    cvll = np.array(infos['cross_val_preds'])
    ll = np.array(infos['ll'])
    print(np.mean(cvll[-1000:]), np.exp(np.mean(cvll[-1000:])))
    # plt.plot(100 * np.exp(cvll))
    # plt.plot(ll[-8000:] / np.mean(ll[-8000:]) * np.mean(cvll[-8000:]))
    # plt.show()
    twelvek_samples.append(np.exp(np.mean(cvll[8000:12000])))
    sixteenk_samples.append(np.exp(np.mean(cvll[12000:])))

    perfs.append(np.exp(np.mean(cvll[-1000:])))
    trial_wise_nll[(mouse_name + "_" + str(fold))] = np.exp(np.mean(cvll[-1000:]))

json.dump(trial_wise_nll, open("ihmm_trialwise_nlls.json", 'w'))