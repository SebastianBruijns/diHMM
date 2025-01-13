import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from mcmc_chain_analysis import state_size_helper, state_num_helper
from dyn_glm_chain_analysis import MCMC_result
from scipy.stats import mannwhitneyu

folder = "./dynamic_GLMiHMM_crossvals/compare_state_nums/"

conditions = [
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
]



# summarise the below code

funcs = []
tresholds = [0.2, 0.1, 0.05, 0.02]
# for t in tresholds:
#     funcs.append(state_num_helper(t))

# no_slow = {t: [] for t in tresholds}
# slow = {t: [] for t in tresholds}

# for file in tqdm(os.listdir(folder)):
#     models = pickle.load(open(folder + file, 'rb'))
#     if models[0].obs_distns[0].Q[0, 0, 0] == 0.0:
#         temp = MCMC_result(models, {'subject': 'meh'}, None, None, 0.0, None)
#         for i, t in enumerate(tresholds):
#             no_slow[t].append(np.mean(funcs[i](temp)))
#     elif models[0].obs_distns[0].Q[0, 0, 0] == 0.04:
#         temp = MCMC_result(models, {'subject': 'meh'}, None, None, 0.04, None)
#         for i, t in enumerate(tresholds):
#             slow[t].append(np.mean(funcs[i](temp)))

# pickle.dump((no_slow, slow), open("state_nums", 'wb'))

no_slow, slow = pickle.load(open("state_nums", 'rb'))

for i, t in enumerate(tresholds):
    print(t, np.mean(no_slow[t]), np.std(no_slow[t]), np.mean(slow[t]), np.std(slow[t]), mannwhitneyu(no_slow[t], slow[t]))
    print(f"effect size {(np.mean(no_slow[t]) - np.mean(slow[t])) / np.std(slow[t])}")
    plt.figure()
    plt.hist(no_slow[t], label='no slow {}'.format(t), alpha=0.4)
    plt.hist(slow[t], label='slow {}'.format(t), alpha=0.4)
    plt.legend()
plt.show()