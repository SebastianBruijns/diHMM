import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

folder = "./dynamic_GLMiHMM_crossvals/infos_lenca/"

cvs = {}

for file in tqdm(os.listdir(folder)):

    infos = json.load(open(folder + file, 'r'))

    subject = infos['subject']
    cv_num = infos['cross_val_num']

    res = np.array(infos['cross_val_preds'])
    res_mean = res[5000:].mean(0)

    if subject not in cvs:
        cvs[subject] = np.zeros((5, res_mean.shape[0]))

    cvs[subject][cv_num] = res_mean

for subject in cvs:
    np.save("./dynamic_GLMiHMM_crossvals/results_lenca/sab_results_{}".format(subject), cvs[subject])