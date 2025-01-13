"""
    Compute which trials go into the same state across samples how often.
    This last bit of info is then also used to create the first plots.
"""
import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6
from dyn_glm_chain_analysis import MCMC_result_list
from dyn_glm_chain_analysis import state_development
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import scipy.cluster.hierarchy as hc

fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][0]
file_prefix = ['.', '/usr/src/app'][0]

subjects = ['KS014']
subjects = [subjects[int(sys.argv[1])]]
fit_variance = [0.04][0]

def state_set_and_plot(test, mode_prefix, subject, fit_type, mode_indices, consistencies):
    # plot the clutering and summary of the fit
    session_bounds = list(np.cumsum([len(s) for s in test.results[0].models[-1].stateseqs]))

    consistencies /= consistencies[0, 0]
    # linkage = hc.linkage(consistencies[0, 0] - consistencies[np.triu_indices(consistencies.shape[0], k=1)], method='complete')
    # pickle.dump(linkage, open(file_prefix + "/temp_linkage_{}.p".format(subject), 'wb'))
    linkage = pickle.load(open(file_prefix + "/temp_linkage_{}.p".format(subject), 'rb'))

    session_bounds = list(np.cumsum([len(s) for s in test.results[0].models[-1].stateseqs]))

    plot_criterion = 0.95

    # hierarchical clustering of trials into states
    a = hc.fcluster(linkage, plot_criterion, criterion='distance')
    
    b, c = np.unique(a, return_counts=1)
    state_sets = []
    for x, y in zip(b, c):
        state_sets.append(np.where(a == x)[0])
    print("dumping state set")
    pickle.dump(state_sets, open(file_prefix + "/multi_chain_saves/{}state_sets_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'wb'))
    
    # plot the ultimate result
    print('temp removal')
    # state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save_append='_{}{}_fitvar_{}'.format(mode_prefix, plot_criterion, fit_variance), show=True, separate_pmf=True, type_coloring=True)


    # code for visualising the consistency matrix and state assignments
    fig, ax = plt.subplots(ncols=5, sharey=True, gridspec_kw={'width_ratios': [10, 1, 1, 1, 1]}, figsize=(13, 8))
    from matplotlib.pyplot import cm
    for j, criterion in enumerate([0.95, 0.8, 0.5, 0.2]):
        clustering_colors = np.zeros((consistencies.shape[0], 100, 4))
        a = hc.fcluster(linkage, criterion, criterion='distance')
        b, c = np.unique(a, return_counts=1)
        print(b.shape)
        print(np.sort(c))

        cmap = cm.rainbow(np.linspace(0, 1, 17))
        rank_to_color_place = dict(zip(range(17), [0, 16, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]))  # handcrafted to maximise color distance
        i = -1
        b = [x for _, x in sorted(zip(c, b))][::-1]
        c = [x for x, _ in sorted(zip(c, b))][::-1]
        plot_above = 50
        while len([y for y in c if y > plot_above]) > 17:
            plot_above += 1
        for x, y in zip(b, c):
            if y > plot_above:
                i += 1
                clustering_colors[a == x] = cmap[rank_to_color_place[i]]

        ax[j+1].imshow(clustering_colors, aspect='auto', origin='upper')
        for sb in session_bounds:
            ax[j+1].axhline(sb, color='k')
        ax[j+1].set_xticks([])
        ax[j+1].set_yticks([])
        ax[j+1].set_title("{}%".format(int(criterion * 100)), size=20)

    ax[0].imshow(consistencies, aspect='auto', origin='upper')
    for sb in session_bounds:
        ax[0].axhline(sb, color='k')
    ax[0].set_xticks([])
    ax[0].set_yticks(session_bounds[::-1])
    ax[0].set_yticklabels(session_bounds[::-1], size=18)
    ax[0].set_ylim(session_bounds[-1], 0)
    ax[0].set_ylabel("Trials", size=28)
    plt.yticks(rotation=45)

    plt.tight_layout()
    plt.savefig("figures/{}clustered_trials_{}_{}".format(mode_prefix, subject, 'criteria comp').replace('.', '_'))
    plt.show()

print(subjects)
for subject in subjects:
    print(subject)
    mode_prefix = 'first_'

    test = pickle.load(open(file_prefix + "/multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'rb'))
    mode_indices = pickle.load(open(file_prefix + "/multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'rb'))
    # consistencies = test.consistency_rsa(indices=mode_indices)  # compute consistency matrix of samples
    # pickle.dump(consistencies, open(file_prefix + "/multi_chain_saves/{}mode_consistencies_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'wb'), protocol=4)
    consistencies = pickle.load(open(file_prefix + "/multi_chain_saves/{}mode_consistencies_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'rb'))
    state_set_and_plot(test, mode_prefix, subject, fit_type, mode_indices=mode_indices, consistencies=consistencies)

    quit()

    # repeat for other modes, if they exist
    mode_prefix = 'second_'
    if os.path.isfile(file_prefix + "/multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance)):
        mode_indices = pickle.load(open(file_prefix + "/multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'rb'))
        consistencies = test.consistency_rsa(indices=mode_indices, mode_prefix=mode_prefix)
        pickle.dump(consistencies, open(file_prefix + "/multi_chain_saves/{}mode_consistencies_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'wb'), protocol=4)
        state_set_and_plot(test, mode_prefix, subject, fit_type, mode_indices=mode_indices, consistencies=consistencies)

    mode_prefix = 'third_'
    if os.path.isfile(file_prefix + "/multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance)):
        mode_indices = pickle.load(open(file_prefix + "/multi_chain_saves/{}mode_indices_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'rb'))
        consistencies = test.consistency_rsa(indices=mode_indices, mode_prefix=mode_prefix)
        pickle.dump(consistencies, open(file_prefix + "/multi_chain_saves/{}mode_consistencies_{}_{}_var_{}.p".format(mode_prefix, subject, fit_type, fit_variance), 'wb'), protocol=4)
        state_set_and_plot(test, mode_prefix, subject, fit_type, mode_indices=mode_indices, consistencies=consistencies)
