"""
    Script for getting data from fits into a state to be analysed

    Script for taking a list of subjects and extracting statistics from the chains
    which can be used to assess which chains have converged to the same regions

"""
import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import pyhsmm
import pickle
import json
from dyn_glm_chain_analysis import MCMC_result
from mcmc_chain_analysis import state_size_helper, state_num_helper, find_good_chains_unsplit_greedy, gamma_func, alpha_func
import index_mice  # executes function for creating dict of available fits
from dyn_glm_chain_analysis import MCMC_result_list
import sys


file_prefix = ['.', '/usr/src/app'][0]  # cluster or local prefix
data_folder = 'dynamic_GLMiHMM_crossvals'
prethinned = None  # will try to determine whether chains where thinned already by thresholding at 2000 samples (since we usually create 4000 per sub-file)

subjects = ['DY_013']
subjects = [subjects[int(sys.argv[1])]]

thinning = 25  # MCMC chain thinning, drop all but every 25th sample

fit_variance = 0.04
fit_type = 'prebias'

# prepare funtions for computing R^hat
func1 = state_num_helper(0.2)
func2 = state_num_helper(0.1)
func3 = state_size_helper()
func4 = state_size_helper(1)
dur = 'yes'

# load file which stores info about all fitted animals
loading_info = json.load(open(file_prefix + "/canonical_infos_fitvar_{}.json".format(fit_variance), 'r'))

def sample_dimensionality_reduction(test, subject, fit_type):
    dim = 3 # number of dimensions to reduce to

    print('Doing PCA')
    ev, eig, projection_matrix, xy, z = test.state_pca(subject, dim=dim)#, save_add="_var_{}".format(fit_variance).replace('.', '_'))
    pickle.dump((xy, z), open(file_prefix + "/multi_chain_saves/xyz_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'wb'))

for subject in subjects:
    
    
    if os.path.isfile(file_prefix + "/multi_chain_saves/" + "canonical_result_{}_prebias_var_0.04.p".format(subject)):
        print("done already")
        # quit()
    assert len(loading_info[subject]["seeds"]) == 16
    assert len(loading_info[subject]["fit_nums"]) == 16

    r_hats = {}
    results = []
    summary_info = {"thinning": thinning, "contains": [], "seeds": [], "fit_nums": []}
    m = len(loading_info[subject]["fit_nums"])  # number of chains
    print(subject)
    n_runs = -1
    chains1 = None

    for j, (seed, fit_num) in enumerate(zip(loading_info[subject]['seeds'], loading_info[subject]['fit_nums'])):
        print(seed)
        samples = []

        mini_counter = 1 # start reading files at 1, discard first 4000 samples (first file) as burnin
        while True:
            try:
                file = file_prefix + "/{}/{}_fittype_{}_var_{}_{}_{}{}.p".format(data_folder, subject, fit_type, fit_variance, seed, fit_num, '_{}'.format(mini_counter))
                samples += pickle.load(open(file, "rb"))
                if prethinned is None:
                    prethinned = len(samples) < 2000  # check whether we already reduced the sample size
                mini_counter += 1
            except Exception:
                print("not found {}".format(file))
                break

        # make sure that all chains have the same number of files (and therefore samples)
        if n_runs == -1:
            n_runs = mini_counter
        else:
            if n_runs != mini_counter:
                print("Problem")
                print(n_runs, mini_counter)
                quit()

        print("num of parts per chain: {}".format(mini_counter))
        save_id = "{}_fittype_{}_var_{}_{}_{}.p".format(subject, fit_type, fit_variance, seed, fit_num).replace('.', '_')


        print("loaded seed {}".format(seed))
        
        if prethinned:
            # we already thinned the chains
            samples = samples
            print(len(samples))
        else:
            samples = samples[::thinning]
        result = MCMC_result(samples,
                            infos={'subject': subject}, data=pickle.load(open(file_prefix + "/{}/{}_fittype_{}_var_{}_{}_{}{}.p".format(data_folder, subject, fit_type, fit_variance, seed, fit_num, '_{}'.format(0)), "rb"))[0].datas,
                             sessions=fit_type, fit_variance=fit_variance,
                             dur=dur, save_id=save_id)
        results.append(result)
        print(result.data is None)

        if chains1 is None:
            n = len(samples)
            chains1 = np.zeros((m, n))
            chains2 = np.zeros((m, n))
            chains3 = np.zeros((m, n))
            chains4 = np.zeros((m, n))

        res = func1(result)
        chains1[j] = res
        res = func2(result)
        chains2[j] = res
        res = func3(result)
        chains3[j] = res
        res = func4(result)
        chains4[j] = res
    
    pickle.dump(chains1, open(file_prefix + "/multi_chain_saves/{}_state_num_0_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains2, open(file_prefix + "/multi_chain_saves/{}_state_num_1_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains3, open(file_prefix + "/multi_chain_saves/{}_largest_state_0_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains4, open(file_prefix + "/multi_chain_saves/{}_largest_state_1_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))

    print()
    print("Checking R^hat, finding best subset of chains")

    sol, final_r_hat = find_good_chains_unsplit_greedy(chains1, chains2, chains3, chains4, reduce_to=8)
    r_hats[subject] = final_r_hat
    loading_info[subject]['ignore'] = sol

    print(r_hats[subject])
    json.dump(loading_info, open(file_prefix + "/multi_chain_saves/canonical_infos_{}_{}_var_{}.json".format(subject, fit_type, fit_variance), 'w'))
    json.dump(r_hats, open(file_prefix + "/multi_chain_saves/canonical_info_r_hats_{}_{}_var_{}.json".format(subject, fit_type, fit_variance), 'w'))

    if r_hats[subject] >= 1.05:
        print("Skipping canonical result due to R^hat")
        continue
    else:
        print("Making canonical result")

    # subset data
    summary_info['contains'] = [i for i in range(m) if i not in sol]
    summary_info['seeds'] = [loading_info[subject]['seeds'][i] for i in summary_info['contains']]
    summary_info['fit_nums'] = [loading_info[subject]['fit_nums'] for i in summary_info['contains']]
    results = [results[i] for i in summary_info['contains']]

    test = MCMC_result_list(results, summary_info)
    pickle.dump(test, open(file_prefix + "/multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'wb'))

    test.r_hat_and_ess(state_num_helper(0.2), False)
    test.r_hat_and_ess(state_num_helper(0.1), False)
    test.r_hat_and_ess(state_num_helper(0.05), False)
    test.r_hat_and_ess(state_size_helper(), False)
    test.r_hat_and_ess(state_size_helper(1), False)
    test.r_hat_and_ess(gamma_func, True)
    test.r_hat_and_ess(alpha_func, True)

    print('Computing sub result')
    sample_dimensionality_reduction(test, subject, fit_type)
