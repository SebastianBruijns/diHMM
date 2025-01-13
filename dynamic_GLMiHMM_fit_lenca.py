"""Start a (series) of iHMM fit(s)."""
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

# test subjects:
file_prefix = ['.', '/usr/src/app'][0]

subjects = ['CSHL045', 'CSHL047', 'CSHL049', 'CSHL051', 'CSHL052', 'CSHL053', 'CSHL054', 'CSHL055', 'CSHL058', 'CSHL059', 'CSHL060', 'CSHL_007', 'CSHL_014', 'CSHL_015',
           'CSHL_020', 'CSH_ZAD_001', 'CSH_ZAD_011', 'CSH_ZAD_017', 'CSH_ZAD_019', 'CSH_ZAD_022', 'CSH_ZAD_024', 'CSH_ZAD_025', 'CSH_ZAD_026', 'CSH_ZAD_029', 'DY_008',
           'DY_009', 'DY_010', 'DY_011', 'DY_013', 'DY_014', 'DY_016', 'DY_018', 'DY_020', 'KS014', 'KS015', 'KS016', 'KS017', 'KS019', 'KS021', 'KS022', 'KS023',
           'KS042', 'KS043', 'KS044', 'KS045', 'KS046', 'KS051', 'KS052', 'KS055', 'KS084', 'KS086', 'KS091', 'KS094', 'KS096', 'MFD_05', 'MFD_06', 'MFD_07', 'MFD_08',
           'MFD_09', 'NR_0017', 'NR_0019', 'NR_0020', 'NR_0021', 'NR_0024', 'NR_0027', 'NR_0028', 'NR_0029', 'NR_0031', 'NYU-06', 'NYU-11', 'NYU-12', 'NYU-27',
           'NYU-30', 'NYU-37', 'NYU-39', 'NYU-40', 'NYU-45', 'NYU-46', 'NYU-47', 'NYU-48', 'NYU-65', 'PL015', 'PL016', 'PL017', 'PL024', 'PL030', 'PL031', 'PL033',
           'PL034', 'PL035', 'PL037', 'PL050', 'SWC_021', 'SWC_022', 'SWC_023', 'SWC_038', 'SWC_039', 'SWC_042', 'SWC_043', 'SWC_052', 'SWC_053', 'SWC_054', 'SWC_058',
           'SWC_060', 'SWC_061', 'SWC_065', 'SWC_066', 'UCLA005', 'UCLA006', 'UCLA011', 'UCLA012', 'UCLA014', 'UCLA015', 'UCLA017', 'UCLA030', 'UCLA033', 'UCLA034',
           'UCLA035', 'UCLA036', 'UCLA037', 'UCLA044', 'UCLA048', 'UCLA049', 'UCLA052', 'ZFM-01576', 'ZFM-01577', 'ZFM-01592', 'ZFM-01935', 'ZFM-01936', 'ZFM-01937',
           'ZFM-02368', 'ZFM-02369', 'ZFM-02370', 'ZFM-02372', 'ZFM-02373', 'ZFM-05236', 'ZM_1897', 'ZM_1898', 'ZM_2240', 'ZM_2241', 'ZM_2245', 'ZM_3003',
           'ibl_witten_13', 'ibl_witten_14', 'ibl_witten_16', 'ibl_witten_17', 'ibl_witten_18', 'ibl_witten_19', 'ibl_witten_20', 'ibl_witten_25', 'ibl_witten_26',
           'ibl_witten_27', 'ibl_witten_29', 'ibl_witten_32']

# subjects = ['KS014']
subjects = ['NYU-07', 'NYU-12', 'CSHL_003', 'NYU-13', 'CSHL_015', 'NYU-04', 'ibl_witten_13', 'ibl_witten_06', 'ibl_witten_12', 'CSHL_014', 'CSHL_010', 'ibl_witten_16', 'CSHL_005', 'CSHL_004', 'NYU-02', 'ibl_witten_15', 'ibl_witten_14', 'CSHL_012', 'IBL-T1', 'IBL-T3', 'IBL-T2', 'NYU-09', 'CSHL_008']

num_subjects = len(subjects)
subjects = [a for a in subjects for i in range(5)] # how often is subject needed, i.e. number of chains or cross-validation folds or seeds for chains
seeds = [100, 101, 102, 103, 104] * num_subjects
cv_nums = [0, 1, 2, 3, 4] * num_subjects


seeds = [seeds[int(sys.argv[1])]]
cv_nums = [cv_nums[int(sys.argv[1])]]
subjects = [subjects[int(sys.argv[1])]]

print(cv_nums)
print(subjects)

for loop_count_i, (subject, cv_num, seed) in enumerate(zip(subjects, cv_nums, seeds)):
    params = {}  # save parameters in a dictionary to save later
    params['subject'] = subject
    params['cross_val_num'] = cv_num
    params['fit_variance'] = 0.04
    params['jumplimit'] = 1
    params['seed'] = seed

    params['file_name'] = file_prefix + "/summarised_sessions_lenca/0_25/{}_prebias_fit_info.csv".format(subject)
    data = pd.read_csv(params['file_name'])
    # save column names
    params['regressors'] = list(data)

    # more obscure params:
    params['gamma_a_0'] = 0.01
    params['gamma_b_0'] = 100
    params['init_var'] = 8
    params['init_mean'] = np.zeros(data.shape[1] - 2)

    r_support = np.arange(5, 705, 4)
    params['dur_params'] = dict(r_support=r_support,
                                r_probs=np.ones(len(r_support))/len(r_support), alpha_0=1, beta_0=1)
    params['alpha_a_0'] = 0.01
    params['alpha_b_0'] = 100
    params['init_state_concentration'] = 3

    params['dur'] = 'yes'

    params['cross_val'] = True
    params['cross_val_type'] = ['normal', 'lenca'][1]
    params['cross_val_fold'] = 10
    params['CROSS_VAL_SEED'] = 4  # Do not change this, it's 4

    params['n_states'] = 15
    params['n_samples'] = 60000
    if params['cross_val']:
        params['n_samples'] = 100
    if params['subject'].startswith("GLM_Sim"):
        print("reduced sample size")
        params['n_samples'] = 48000

    print(params['n_samples'])

    # find a unique identifier to save this fit
    while True:
        folder = file_prefix + "/dynamic_GLMiHMM_crossvals_2/"
        rand_id = np.random.randint(1000)
        if params['cross_val']:
            id = "{}_crossval_{}_{}_var_{}_{}".format(params['subject'], params['cross_val_num'],
                                                         params['fit_variance'], params['seed'], rand_id)
        else:
            id = "{}_fittype_prebias_var_{}_{}_{}".format(params['subject'],
                                                     params['fit_variance'], params['seed'], rand_id)
        if not os.path.isfile(folder + id + '_0.p'):
            break
    # create placeholder dataset for rand_id purposes
    pickle.dump(params, open(folder + id + '_0.p', 'wb'))

    print(id)
    np.random.seed(params['seed'])

    # check that the data file contains all session numbers from 0 the the max
    assert np.all(np.unique(data['session']) == np.arange(np.max(data['session']) + 1)), "Data file does not contain all session numbers"

    T = int(data['session'].max() + 1)
    n_inputs = data.shape[1] - 2

    # set up model
    obs_hypparams = {'n_regressors': n_inputs, 'T': T, 'jumplimit': params['jumplimit'], 'prior_mean': params['init_mean'],
                        'P_0': params['init_var'] * np.eye(n_inputs), 'Q': params['fit_variance'] * np.tile(np.eye(n_inputs), (T, 1, 1))}
    obs_distns = [distributions.Dynamic_GLM(**obs_hypparams) for state in range(params['n_states'])]
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

    # ingest data, possibly setting up cross-validation
    if params['cross_val']:
        rng = np.random.RandomState(params['CROSS_VAL_SEED'])

    lenca_counter = 0
    data_save = []
    for session in range(T):

        session_data = data[data['session'] == session].values
        data_save.append(session_data[:, 1:].copy())

        if params['cross_val']:
            if params['cross_val_type'] == 'normal':
                test_sets = np.tile(np.arange(params['cross_val_fold']), session_data.shape[0] // params['cross_val_fold'] + 1)[:session_data.shape[0]]
                rng.shuffle(test_sets)
                session_data[(test_sets == params['cross_val_num']).astype(bool), -1] = None
            elif params['cross_val_type'] == 'lenca':
                lenca_info = np.load(file_prefix + "/lenca_data/" + "{}_data_and_indices_CV_5_folds.npz".format(params['subject']))
                trials_to_nan = lenca_info['presentTest'][params['cross_val_num']][lenca_counter:lenca_counter + session_data.shape[0]]
                assert trials_to_nan.shape[0] == (lenca_info['sessInd'][session+1] - lenca_info['sessInd'][session])
                lenca_counter += session_data.shape[0]
                session_data[trials_to_nan.astype(bool), -1] = None
            else:
                print('Incorrectly specified crossvalidation type')
                quit()

        posteriormodel.add_data(session_data[:, 1:])

    # resample model
    time_save = time.time()
    likes = np.zeros(params['n_samples'])
    cross_val_lls = []
    models = []
    with warnings.catch_warnings():  # ignore the scipy warning
        warnings.simplefilter("ignore")
        for j in range(params['n_samples']):

            if j % 40 == 0 or j == 3:
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

            # save unfinished results
            if j % 20 == 0 and j > 0:
                if params['n_samples'] <= 40:
                    pickle.dump(models, open(folder + id + '.p', 'wb'))
                else:
                    pickle.dump(models, open(folder + id + '_{}.p'.format(j // 4001), 'wb'))
                    if j % 40 == 0:
                        cross_val_lls.append(eval_cross_val(models, copy.deepcopy(posteriormodel.datas), data_save, n_all_states=params['n_states']))
                        models = []
    print(time.time() - time_save)

    # save info
    if params['cross_val']:
        cross_val_lls.append(eval_cross_val(models, copy.deepcopy(posteriormodel.datas), data_save, n_all_states=params['n_states']))
        cross_val_lls = np.concatenate(cross_val_lls)
        lls_mean = np.mean(cross_val_lls[-10:])
        params['cross_val_preds'] = [x.tolist() for x in cross_val_lls]

    print(id)
    params['dur_params']['r_support'] = params['dur_params']['r_support'].tolist()
    params['dur_params']['r_probs'] = params['dur_params']['r_probs'].tolist()
    params['ll'] = likes.tolist()
    params['init_mean'] = params['init_mean'].tolist()
    if params['cross_val']:
        json.dump(params, open(folder + "infos/" + '{}_{}_cvll_{}_{}_{}_{}.json'.format(params['subject'], params['cross_val_num'], str(np.round(lls_mean, 3)).replace('.', '_'),
                                                                                               params['fit_variance'], params['seed'], rand_id), 'w'))
    else:
        json.dump(params, open(folder + "infos/" + '{}_{}_{}_{}.json'.format(params['subject'], params['fit_variance'], params['seed'], rand_id), 'w'))
    pickle.dump(models, open(folder + id + '_{}.p'.format(j // 4001), 'wb'))
