"""
    Start a (series) of diHMM fit(s).
    Expects to be called with an integer as console argument, specifying which chain to start sampling (start with 0)
    This currently saves every single sample, for completeness, even though later we only analyse every 25th sample. The memory and runtime footprint can be reduced by only saving every 25th sample here in the first place.
    Make sure to create the folders specified on lines 43-45
"""
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import pyhsmm
import pyhsmm.basic.distributions as distributions
import copy
import warnings
import pickle
import time
import numpy as np
import json
import sys
import pandas as pd

def eval_cross_val(models, data, unmasked_data, n_all_states):
    """Eval cross_validation performance on held-out datapoints of an instantiated model"""
    lls = np.zeros((len(models), len(data)))
    cross_val_n = np.zeros(len(data))
    for sess_time, (d, full_data) in enumerate(zip(data, unmasked_data)):
        held_out = np.isnan(d[:, -1])
        cross_val_n[sess_time] += held_out.sum()
        d[:, -1][held_out] = full_data[:, -1][held_out]
        for i, m in enumerate(models):
            for s in range(n_all_states):
                mask = np.logical_and(held_out, m.stateseqs[sess_time] == s)
                if mask.sum() > 0:
                    ll = m.obs_distns[s].log_likelihood(d[mask], sess_time)
                    lls[i, sess_time] += np.sum(ll)
    lls /= cross_val_n
    return lls

# specify folders and subjects to fit
file_prefix = '.'
data_folder = file_prefix + "/summarised_sessions/0_25/"
output_folder = file_prefix + "/dynamic_GLMiHMM_crossvals/"
output_infos = output_folder + "infos/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_infos, exist_ok=True)


subjects = ['KS014']

# creat lists of seeds (and cross-validation numbers if needed) for the subjects, we used 16 chains per subject
num_subjects = len(subjects)
subjects = [a for a in subjects for i in range(16)] # how often is subject needed, i.e. number of chains or cross-validation folds or seeds for chains
seeds = list(range(200, 217)) * num_subjects
cv_nums = [0] * num_subjects * 16  # relevant only when doing cross-validation, specifies the fold to hold out

seeds = [seeds[int(sys.argv[1])]]
cv_nums = [cv_nums[int(sys.argv[1])]]
subjects = [subjects[int(sys.argv[1])]]

print(cv_nums)
print(subjects)

for subject, cv_num, seed in zip(subjects, cv_nums, seeds):

    ### Parameter setting

    params = {}  # save parameters in a dictionary to save later
    params['subject'] = subject
    params['cross_val_num'] = cv_num
    params['seed'] = seed
    params['file_name'] = data_folder + "{}_prebias_fit_info.csv".format(subject)

    # read in data-frame, expects the following format of columns: "session", [regressor 1], ..., [regressor n], "choice"
    data = pd.read_csv(params['file_name'])
    # save column names
    params['regressors'] = list(data)

    # hyper-parameters of observation distribution (the type of observation distribution is specified later, we used distributions.Dynamic_GLM)
    params['fit_variance'] = 0.04
    params['init_var'] = 8
    params['init_mean'] = np.zeros(data.shape[1] - 2)

    # iHMM transition matrix parameters
    params['gamma_a_0'] = 0.01
    params['gamma_b_0'] = 100
    params['alpha_a_0'] = 0.01
    params['alpha_b_0'] = 100
    params['init_state_concentration'] = 3
    params['n_states'] = 15  # L of the weak-limit approximation, maximum number of states to infer

    # duration distribution parameters
    params['dur'] = 'yes'
    r_support = np.arange(5, 705)
    params['dur_params'] = dict(r_support=r_support,
                                r_probs=np.ones(len(r_support))/len(r_support), alpha_0=1, beta_0=1)

    # special stuff
    params['jumplimit'] = 1  # whether states can change dynamically during sessions in which they are not present. We always used 1, prohibiting such changes

    # specify cross-validation, if desired
    params['cross_val'] = False
    params['cross_val_fold'] = 10
    params['CROSS_VAL_SEED'] = 4  # Do not change this, it's 4

    # number of samples to draw    
    params['n_samples'] = 48000
    if params['cross_val']:
        params['n_samples'] = 12000

    ### Parameters set

    ### find a unique identifier to save this fit

    while True:
        rand_id = np.random.randint(1000)
        if params['cross_val']:
            id = "{}_crossval_{}_{}_var_{}_{}".format(params['subject'], params['cross_val_num'],
                                                         params['fit_variance'], params['seed'], rand_id)
        else:
            id = "{}_fittype_prebias_var_{}_{}_{}".format(params['subject'],
                                                     params['fit_variance'], params['seed'], rand_id)
        if not os.path.isfile(output_folder + id + '_0.p'):
            break
    # create placeholder dataset for rand_id purposes
    pickle.dump(params, open(output_folder + id + '_0.p', 'wb'))

    # check that the data file contains all session numbers from 0 the the max
    assert np.all(np.unique(data['session']) == np.arange(np.max(data['session']) + 1)), "Data file does not contain all session numbers"

    ### set up model

    np.random.seed(params['seed'])
    T = int(data['session'].max() + 1)  # number of time points (sessions in our case)
    n_inputs = data.shape[1] - 2  # number of input variables, number of rows of dataframe, -2 for session and choice column
    obs_hypparams = {'n_regressors': n_inputs, 'T': T, 'jumplimit': params['jumplimit'], 'prior_mean': params['init_mean'],
                        'P_0': params['init_var'] * np.eye(n_inputs), 'Q': params['fit_variance'] * np.tile(np.eye(n_inputs), (T, 1, 1))}
    obs_distns = [distributions.Dynamic_GLM(**obs_hypparams) for state in range(params['n_states'])]  # can chose a different observation distribution here
    dur_distns = [distributions.NegativeBinomialIntegerR2Duration(**params['dur_params']) for state in range(params['n_states'])]

    if params['dur'] == 'yes':
        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                # https://math.stackexchange.com/questions/449234/vague-gamma-prior
                alpha_a_0=params['alpha_a_0'], alpha_b_0=params['alpha_b_0'],  # gamma steers state number
                gamma_a_0=params['gamma_a_0'], gamma_b_0=params['gamma_b_0'],
                init_state_concentration=params['init_state_concentration'],
                obs_distns=obs_distns,
                dur_distns=dur_distns,
                var_prior=params['fit_variance'])
    else:
        posteriormodel = pyhsmm.models.WeakLimitHDPHMM(
                alpha_a_0=params['alpha_a_0'], alpha_b_0=params['alpha_b_0'],
                gamma_a_0=params['gamma_a_0'], gamma_b_0=params['gamma_b_0'],
                init_state_concentration=params['init_state_concentration'],
                obs_distns=obs_distns,
                var_prior=params['fit_variance'])

    ### ingest data, possibly setting up cross-validation
    
    if params['cross_val']:
        rng = np.random.RandomState(params['CROSS_VAL_SEED'])

    data_save = []
    for session in range(T):

        session_data = data[data['session'] == session].values
        data_save.append(session_data[:, 1:].copy())

        if params['cross_val']:
            test_sets = np.tile(np.arange(params['cross_val_fold']), session_data.shape[0] // params['cross_val_fold'] + 1)[:session_data.shape[0]]
            rng.shuffle(test_sets)
            session_data[(test_sets == params['cross_val_num']).astype(bool), -1] = None

        posteriormodel.add_data(session_data[:, 1:])

    ### MCMC sampling

    time_save = time.time()
    likes = np.zeros(params['n_samples'])
    cross_val_lls = []
    models = []
    with warnings.catch_warnings():  # ignore the scipy warning
        warnings.simplefilter("ignore")
        for j in range(params['n_samples']):

            if j % 400 == 0 or j == 3:
                print(j)

            posteriormodel.resample_model()

            likes[j] = posteriormodel.log_likelihood()
            model_save = copy.deepcopy(posteriormodel)
            if j != params['n_samples'] - 1 and j != 0:
                # To save on memory we delete the data from all but the first and last model
                model_save.delete_data()
                model_save.delete_obs_data()
                if params['dur'] == 'yes':
                    model_save.delete_dur_data()
            models.append(model_save)

            if params['cross_val']:
                cross_val_lls.append(eval_cross_val(models, copy.deepcopy(posteriormodel.datas), data_save, n_all_states=params['n_states']))
            
            # save unfinished results
            if j % 2000 == 0 and j > 0:
                if params['n_samples'] <= 4000:
                    pickle.dump(models, open(output_folder + id + '.p', 'wb'))
                else:
                    pickle.dump(models, open(output_folder + id + '_{}.p'.format(j // 4001), 'wb'))
                    if j % 4000 == 0:
                        models = []
    print(time.time() - time_save)

    ### save info

    if params['cross_val']:
        lls_mean = np.mean(cross_val_lls[-1000:])
        params['cross_val_preds'] = cross_val_lls.tolist()

    print(id)
    params['dur_params']['r_support'] = params['dur_params']['r_support'].tolist()  # turn into list to save via json
    params['dur_params']['r_probs'] = params['dur_params']['r_probs'].tolist()
    params['ll'] = likes.tolist()
    params['init_mean'] = params['init_mean'].tolist()
    if params['cross_val']:
        json.dump(params, open(output_infos + '{}_{}_cvll_{}_{}_{}_{}.json'.format(params['subject'], params['cross_val_num'], str(np.round(lls_mean, 3)).replace('.', '_'),
                                                                                               params['fit_variance'], params['seed'], rand_id), 'w'))
    else:
        json.dump(params, open(output_infos + '{}_{}_{}_{}.json'.format(params['subject'], params['fit_variance'], params['seed'], rand_id), 'w'))
    pickle.dump(models, open(output_folder + id + '_{}.p'.format(j // 4001), 'wb'))
