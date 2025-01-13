"""
    This script is used to select samples from a mode, out of all the samples of the used MCMC chains

    The code will ask to "Pick level", at which point one types in a lower threshold for the estimated sample probability (of figure "PCA density [name] (3 dim) prebias.png"), e.g.:
    5e-10
    Alternatively, if it is necessary to separate out different modes from the posterior, one can type "cond" (for adding conditions), which will cause the code
    to ask for lower and upper thresholds along all three PC dimensions (x, y, z), followed by another request for a lower density estimate limit.

    Given these, we compute how many samples fulfull the set conditions and their extent along all three PC dimensions. This can be used to ensure that an appropriate
    number of samples is selected which is purely within a single desired mode. This can be confirmed by typing "yes" when asked whether one is happy, we typically aim for just over 400 samples.

    The code then asks "Subset by factor?", which can be used for highly concentrated modes which are hard to restrict to a sensible number of samples via a lower threshold.
    Respond with "yes" or "y", then give an integer n, which subsets the colletion to every nth sample which fulfills the listed conditions.

    Lastly, it presents the option to isolate another mode (if there's a relevant second or third mode), which can be responded with "yes"/"y" if the spiel is to be repeated.
"""
import json
import numpy as np
from dyn_glm_chain_analysis import MCMC_result_list, MCMC_result
import pickle
import matplotlib.pyplot as plt


fit_variance = 0.04
subjects = ["KS014"]

def create_mode_indices(test, subject, fit_type):
    """
        This code, together with the figures produced in part 1, is used to select a collection of samples corresponding to 1 or more modes of the posterior
    """
    xy, z = pickle.load(open("multi_chain_saves/xyz_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'rb'))

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
        Find a good threshold for taking samples, possibly take into account thresholds from the PC dimensions.
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

fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][0]

for subject in subjects:
    try:
        test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'rb'))

        info_dict = pickle.load(open("./session_data/{}_info_dict.p".format(subject), "rb"))
        test.results[0].infos = info_dict  # posthoc adding of info
        test.results[0].n_contrasts = 11
        pickle.dump(test, open("multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'wb'))
        print('Computing sub result')
        create_mode_indices(test, subject, fit_type)
    except:
        continue
