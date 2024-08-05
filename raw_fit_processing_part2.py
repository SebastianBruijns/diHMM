"""
    This script is used to select samples from a mode, out of all the samples of the used MCMC chains 
"""
import json
import numpy as np
from dyn_glm_chain_analysis import MCMC_result_list, MCMC_result
import pickle
import matplotlib.pyplot as plt
import os


fit_variance = 0.03
subjects = ['GLM_Sim_13', 'fip_29', 'fip_16', 'GLM_Sim_12']

def create_mode_indices(test, subject, fit_type):
    """
        This code, together with the figures produced in part 1, is used to select a collection of samples corresponding to 1 or more modes of the posterior
    """
    try:
        xy, z = pickle.load(open("multi_chain_saves/xyz_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'rb'))
    except Exception as e:
        print(e)
        # this is cluster work
        return

    print("Mode indices of " + subject)

    threshold_search(xy, z, test, 'first_', subject, fit_type)

    print("Find another mode?")
    if input() not in ['yes', 'y']:
        return

    threshold_search(xy, z, test, 'second_', subject, fit_type)

    print("Find another mode?")
    if input() not in ['yes', 'y']:
        return

    threshold_search(xy, z, test, 'third_', subject, fit_type)
    return


def threshold_search(xy, z, test, mode_prefix, subject, fit_type):
    """
        Find a good threshold for taking samples
    """
    happy = False
    conds = [0, None, None, None, None, None, None]
    x_min, x_max, y_min, y_max, z_min, z_max = None, None, None, None, None, None
    while not happy:
        print()
        print("Pick level")
        prob_level = input()
        if prob_level == 'cond':
            print("x > ?")
            resp = input()
            if resp not in ['n', 'no']:
                x_min = float(resp)
            else:
                x_min = None

            print("x < ?")
            resp = input()
            if resp not in ['n', 'no']:
                x_max = float(resp)
            else:
                x_max = None

            print("y > ?")
            resp = input()
            if resp not in ['n', 'no']:
                y_min = float(resp)
            else:
                y_min = None

            print("y < ?")
            resp = input()
            if resp not in ['n', 'no']:
                y_max = float(resp)
            else:
                y_max = None

            print("z > ?")
            resp = input()
            if resp not in ['n', 'no']:
                z_min = float(resp)
            else:
                z_min = None

            print("z < ?")
            resp = input()
            if resp not in ['n', 'no']:
                z_max = float(resp)
            else:
                z_max = None

            print("Prob level")
            prob_level = float(input())
            conds = [prob_level, x_min, x_max, y_min, y_max, z_min, z_max]
            print("Condtions are {}".format(conds))
        else:
            try:
                prob_level = float(prob_level)
            except:
                print('mistake')
                prob_level = float(input)
            conds[0] = prob_level

        print("Level is {}".format(prob_level))

        mode = conditions_fulfilled(z, xy, conds)
        print("# of samples: {}".format(mode.sum()))
        mode_indices = np.where(mode)[0]
        if mode.sum() > 0:
            print(xy[0][mode_indices].min(), xy[0][mode_indices].max(), xy[1][mode_indices].min(), xy[1][mode_indices].max(), xy[2][mode_indices].min(), xy[2][mode_indices].max())
            print("Happy?")
            happy = 'yes' == input()

    print("Subset by factor?")
    if input() in ['yes', 'y']:
        print("Factor?")
        print(mode_indices.shape)
        factor = int(input())
        mode_indices = mode_indices[::factor]
        print(mode_indices.shape)
    if subject not in loading_info:
        loading_info[subject] = {}
    loading_info[subject]['mode prob level'] = prob_level

    pickle.dump(mode_indices, open("multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'wb'))
    # rest is on the cluster now


def conditions_fulfilled(z, xy, conds):
    # check whether a sample, as projected onto PC space, fulfills all desired conditions
    works = z > conds[0]
    if conds[1] is not None:
        works = np.logical_and(works, xy[0] > conds[1])
    if conds[2] is not None:
        works = np.logical_and(works, xy[0] < conds[2])
    if conds[3] is not None:
        works = np.logical_and(works, xy[1] > conds[3])
    if conds[4] is not None:
        works = np.logical_and(works, xy[1] < conds[4])
    if conds[5] is not None:
        works = np.logical_and(works, xy[2] > conds[5])
    if conds[6] is not None:
        works = np.logical_and(works, xy[2] < conds[6])

    return works

fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][2]
if fit_type == 'bias':
    loading_info = json.load(open("canonical_infos_bias_fitvar_{}.json".format(fit_variance), 'r'))
elif fit_type == 'prebias':
    loading_info = json.load(open("canonical_infos_fitvar_{}.json".format(fit_variance), 'r'))
elif fit_type == 'all':
    loading_info = json.load(open("canonical_infos_all_fitvar_{}.json".format(fit_variance), 'r'))

for subject in subjects:
    test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'rb'))
    # if os.path.isfile("multi_chain_saves/{}mode_indices_{}_{}.p".format('first_', subject, fit_type)):
    #     print("It has been done")
    #     continue
    print('Computing sub result')
    create_mode_indices(test, subject, fit_type)
