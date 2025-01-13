import matplotlib
import os
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import matplotlib.pyplot as plt
import pyhsmm
import pickle
import seaborn as sns
import sys
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import combinations, product
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json
import multiprocessing as mp
from mcmc_chain_analysis import state_num_helper, ll_func, r_hat_array_comp, rank_inv_normal_transform
import pandas as pd
from analysis_pmf import pmf_type, type2color
import re
import datetime
import time
from scipy.stats import gaussian_kde

performance_points = np.array([-1, -1, 0, 0])

def pmf_to_perf(pmf):
    # determine performance of a pmf, but only on the omnipresent strongest contrasts
    return np.mean(np.abs(performance_points + pmf[[0, 1, -2, -1]]))

#colors = np.genfromtxt('colors.csv', delimiter=',')

np.set_printoptions(suppress=True)
file_prefix = ['.', '/usr/src/app'][0]

fs = 16
num_to_cont = dict(zip(range(11), [-1., -0.5, -.25, -.125, -.062, 0., .062, .125, .25, 0.5, 1.]))
contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
cont_mapping = np.vectorize(contrast_to_num.get)

all_conts = np.array([-1, -0.5, -.25, -.125, -.062, 0, .062, .125, .25, 0.5, 1])
all_cont_ticks = (np.arange(11), [-1, -0.5, -.25, -.125, -.062, 0, .062, .125, .25, 0.5, 1])
bias_cont_ticks = (np.arange(9), [-1, -.25, -.125, -.062, 0, .062, .125, .25, 1])

contrasts_L = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])
contrasts_R = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])[::-1]


def weights_to_pmf(weights, with_bias=1):
    psi = weights[0] * contrasts_R + weights[1] * contrasts_L + with_bias * weights[-1]
    return 1 / (1 + np.exp(psi))  # we somehow got the answers twisted, so we drop the minus here to get the opposite response probability for plotting



class MCMC_result_list:
    """
        Class to combine multiple MCMC results instances and functions to extract info or plot the posterior.
    """
    def __init__(self, results, summary_info):
        self.results = results
        self.summary_info = summary_info
        self.n = self.results[0].n_samples
        self.m = len(self.results)

    def r_hat_and_ess(self, func, model_for_loop, rank_norm=True, mode_indices=None):
        """Compute all kinds of R^hat's and the effective sample size with the intermediate steps
           Following Gelman page 284f"""

        if mode_indices is None:
            chains = np.zeros((self.m, self.n))
            if model_for_loop:
                for j, result in enumerate(self.results):
                    for i in range(self.n):
                        chains[j, i] = func(result.models[i])
            else:
                for i in range(self.m):
                    chains[i] = func(self.results[i])
        else:
            inds = []
            for lims in [range(i * self.n, (i + 1) * self.n) for i in range(self.m)]:
                inds.append([ind - lims[0] for ind in mode_indices if ind in lims])
            lens = [len(ind) for ind in inds]
            min_len = min([li for li in lens if li != 0])
            n_remaining_chains = len([li for li in lens if li != 0])
            self.m, self.n = n_remaining_chains, min_len
            print("{} chains left with a len of {}".format(n_remaining_chains, min_len))
            chains = np.zeros((n_remaining_chains, min_len))
            counter = -1
            for i, ind in enumerate(inds):
                if len(ind) == 0:
                    continue
                counter += 1
                step = len(ind) // min_len
                up_to = min_len * step
                chains[counter] = func(self.results[i], ind[:up_to:step])

        self.chains = chains
        self.rank_normalised, self.folded_rank_normalised, self.ranked, self.folded_ranked = rank_inv_normal_transform(self.chains)

        # Compute all R^hats, use the worst
        self.lame_r_hat, self.lame_var_hat_plus = r_hat_array_comp(self.chains)
        self.rank_normalised_r_hat, self.rank_normed_var_hat_plus = r_hat_array_comp(self.rank_normalised)
        self.folded_rank_normalised_r_hat, self.folded_rank_normalised_var_hat_plus = r_hat_array_comp(self.folded_rank_normalised)
        self.r_hat = max(self.lame_r_hat, self.rank_normalised_r_hat, self.folded_rank_normalised_r_hat)
        print("r_hat is {:.4f} (max of normal ({:.4f}), rank normed ({:.4f}), folded rank normed ({:.4f}))".format(self.r_hat, self.lame_r_hat, self.rank_normalised_r_hat, self.folded_rank_normalised_r_hat))

        t = 1
        rhos = []
        # use chains and var_hat_plus as desired to compute effective sample size
        local_chains = self.rank_normalised if rank_norm else self.chains
        local_var_hat_plus = self.rank_normed_var_hat_plus if rank_norm else self.lame_var_hat_plus

        local_var_hat_plus = max(local_var_hat_plus, 0.00001)  # avoid division by zero

        # Estimate sample auto-correlation (could be done with Fourier, but Gelman doesn't elaborate)
        while True:
            V_t = np.sum((local_chains[:, t:] - local_chains[:, :-t]) ** 2)
            rho_t = 1 - (V_t / (self.m * (self.n - t))) / (2 * local_var_hat_plus)
            t += 1
            rhos.append(rho_t)
            if (t > 2 and t % 2 == 1 and rhos[-1] + rhos[-2] < 0) or t == self.n:
                break
        self.n_eff = self.m * self.n / (1 + 2 * sum(rhos[:-2]))
        print("Effective number of samples is {}".format(self.n_eff))

        return chains

    def histos(self, type='normal'):
        """Plot histograms of different way to represent the features
           TODO: It would also be interesting to look at all mode_indices, not just the ones left after reduction
           Using 'ranked' gives Gelman's beloved rank histograms
           Need to use r_hat_and_ess before this to initialise chain variables with a feature
        """
        if type == 'normal':
            local_chain = self.chains
        elif type == 'folded_rank':
            local_chain = self.folded_ranked
        elif type == 'ranked':
            local_chain = self.ranked
        count_max = 0
        for i in range(self.m):
            counts, _ = np.histogram(local_chain[i])
            count_max = max(count_max, counts.max())
        _, bins = np.histogram(local_chain)
        for i in range(self.m):
            plt.subplot(self.m // 2, 2, i+1)
            plt.hist(local_chain[i], bins=bins)
            plt.ylim(top=count_max)
        plt.show()

    def consistency_rsa(self, plot=True, indices=[], mode_prefix=""):
        # We compute the consistency of state assigments across samples, this will later allow us to cluster trials into states
        # take trial 30 and trial 3700 (say). We have n iHMM samples - and an
        # assignment of each trial to a state in each sample.
        # Use an RSA matrix whose (30,3700) entry is the fraction of the n
        # samples for which those states are the same (or different).

        consistency_mat = np.zeros((self.results[0].n_datapoints, self.results[0].n_datapoints))
        for i, m in enumerate([item for sublist in self.results for item in sublist.models]):  # for each sample across all chains
            if i not in indices:
                continue
            states = np.concatenate(m.stateseqs)  # flatten all sequences into one
            for s in range(self.results[0].n_all_states):       # for all states
                finds = np.where(states == s)[0]                # find where it is active
                consistency_mat[tuple(np.meshgrid(finds, finds))] += 1  # mark all trials pairs where a state co-occurs
        if plot:
            plt.imshow(consistency_mat)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(file_prefix + "/dynamic_GLM_figures/" + mode_prefix + "consistency_matrix_{}.png".format(self.results[0].save_id))
            # plt.show()
            plt.close()
        return consistency_mat

    def state_pca(self, subject, dim=2):
        if False:
            # load, if this is already available
            xy, z = pickle.load(open(file_prefix + "/multi_chain_saves/xyz_{}_{}.p".format(subject, 'prebias'), 'rb'))
        else:
            dist_vectors = []
            for result in self.results:  # iterate over all samples
                temp = time.time()
                a = result.state_rsa(result.n_datapoints // 200)  # divide datapoints in ~170 bins
                print(time.time() - temp)

                # flatten and append the matrix
                # a = a.reshape(a.shape[0], a.shape[1] ** 2)
                dist_vectors.append(a)
            pca_vecs = np.concatenate(dist_vectors)

            print(pca_vecs.shape)
            cov = np.cov(pca_vecs.T)
            ev, eig = np.linalg.eigh(cov)
            ev = np.flip(ev)
            eig = np.flip(eig, axis=1)
            # ev, eig = np.linalg.eigh(cov)
            # eig = eig[:, np.argsort(-ev)]
            # ev = ev[:, np.argsort(-ev)]
            ev = ev / ev.sum()
            print(ev[:10])
            projection_matrix = eig[:, :dim].T
            dimreduc = np.real(projection_matrix.dot(pca_vecs.T))

            # get reduced coordinates and perform density estimation
            xy = np.vstack([dimreduc[i] for i in range(dim)])
            z = gaussian_kde(xy)(xy)

        # plot some useful figures for finding the mode of the posterior
        plt.figure(figsize=(16, 9))
        if dim == 2:
            plt.scatter(dimreduc[0], dimreduc[1], c=z, cmap='inferno')
            plt.xlabel("PC 1", size=36)
            plt.ylabel("PC 2", size=36)
            plt.gca().spines[['right', 'top']].set_visible(False)
        else:
            plt.subplot(1, 3, 1)
            plt.scatter(dimreduc[0], dimreduc[1], c=z)
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            official_xlims = plt.xlim()
            official_ylims = plt.ylim()

            plt.subplot(1, 3, 2)
            plt.scatter(dimreduc[0], dimreduc[2], c=z)
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 3")

            plt.subplot(1, 3, 3)
            plt.scatter(dimreduc[1], dimreduc[2], c=z)
            plt.xlabel("Dim 2")
            plt.ylabel("Dim 3")
        cbar = plt.colorbar()
        cbar.set_label("Density estimate", rotation=270, size=30, labelpad=35)
        plt.title("Expl. variance of first {} PCs: {}".format(dim, ev[:dim].sum()))
        plt.tight_layout()
        plt.savefig(file_prefix + "/dynamic_GLM_figures/PCA density {} ({} dim) {}".format(subject, dim, self.results[0].type))
        plt.show()

        chains = dimreduc[0].reshape(pca_vecs.shape[0] // self.n, self.n)
        r_hat, _ = r_hat_array_comp(chains)
        plt.figure(figsize=(16, 9))
        for i in range(pca_vecs.shape[0] // self.n):
            plt.scatter(dimreduc[0, i * self.n: (i+1) * self.n], dimreduc[1, i * self.n: (i+1) * self.n])
        plt.title("R^hat of first PC coord. = {}".format(r_hat))
        plt.xlim(official_xlims)
        plt.ylim(official_ylims)
        plt.tight_layout()
        plt.savefig(file_prefix + "/dynamic_GLM_figures/Colorful chains {} {}".format(subject, self.results[0].type))
        plt.close()

        fig = plt.figure(figsize=(16, 9))
        n_rows = 2 if len(self.results) == 8 else 4
        for i in range(pca_vecs.shape[0] // self.n):
            plt.subplot(n_rows, 4, i+1)
            im = plt.scatter(dimreduc[0, i * self.n: (i+1) * self.n], dimreduc[1, i * self.n: (i+1) * self.n], c=np.arange(self.n))
            plt.xticks([])
            plt.yticks([])
            plt.xlim(official_xlims)
            plt.ylim(official_ylims)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(file_prefix + "/dynamic_GLM_figures/Individual chains {} {}".format(subject, self.results[0].type))
        plt.close()

        return ev, eig, projection_matrix, xy, z


class MCMC_result:
    def __init__(self, models, infos, data, sessions, fit_variance, save_id, dur='yes', sample_lls=None):

        self.models = models
        self.name = infos['subject']
        self.type = sessions
        self.fit_variance = str(fit_variance).replace('.', '_')
        self.save_id = save_id
        self.infos = infos
        self.data = data

        self.n_samples = len(self.models)
        self.n_sessions = len(self.models[-1].stateseqs)
        self.n_datapoints = sum([len(s) for s in self.models[-1].stateseqs])
        self.n_all_states = self.models[-1].num_states
        self.sample_lls = sample_lls

        self.cont_ticks = bias_cont_ticks

        # self.session_contrasts = [np.unique(cont_mapping(d[:, 0] - d[:, 1])) for d in self.data]

        self.count_assigns()

    def count_assigns(self):
        self.assign_counts = np.zeros((self.n_samples, self.n_all_states))
        for i, m in enumerate(self.models):
            flat_list = np.array([item for sublist in m.stateseqs for item in sublist])

            for s in range(self.n_all_states):
                self.assign_counts[i, s] = np.sum(flat_list == s)

    def state_rsa(self, trials_per_bin):
        # perform rsa across samples: bin trials in bins of size trials_per_bin, compute a dissimilarity matrix witin one sample
        # do this for every sample, then compare the dissim. matrices

        dist_matrices = []
        for m in self.models:
            # first, package all the state samples in sets of trials_per_bin
            state_hists = []  # histograms over state usage 
            seq_num = 0  # which session are we currently considering
            seq_total = 0  # will count how many trials of the current session we have considered already
            leftovers = []  # last few trials of a previous session (weren't enough to fill a bin)
            while (seq_num != self.n_sessions - 1) or (len(m.stateseqs[-1]) > seq_total + trials_per_bin):  # while not all trials are binned up
                if len(leftovers) + len(m.stateseqs[seq_num][seq_total:]) < trials_per_bin:
                    # if the trials remaining in the current session aren't enough...
                    # ... just put them in the leftover list, increase the counter and set the current total to 0
                    leftovers += list(m.stateseqs[seq_num][seq_total:])
                    seq_num += 1
                    seq_total = 0
                else:
                    # if there are still enough trials in the current session, compute the histogram (possibly taking into account previous leftovers)
                    state_hists.append(np.bincount(leftovers + list(m.stateseqs[seq_num][seq_total:seq_total + trials_per_bin - len(leftovers)]), minlength=self.n_all_states))
                    seq_total += trials_per_bin - len(leftovers)
                    # clear leftover
                    leftovers = []

            # compute matrix of distances between dist.s (this simply reduces to the absolute difference over state counts)
            n_bins = len(state_hists)
            dist_matrix = np.zeros((int(n_bins * (n_bins - 1) / 2)))
            for counter, (i, j) in enumerate(np.nditer(np.triu_indices(n_bins, k=1))):
                dist_matrix[counter] = np.sum(np.abs(state_hists[i] - state_hists[j]))

            # append all the distance matrices
            # TODO: this is a waste of matrix space, it's just symmetric!
            # dist_matrix += dist_matrix.T
            dist_matrices.append(dist_matrix)
        return np.array(dist_matrices)

    def trial_glm_weights(self):
        glm_weights = []
        for m in self.models:
            temp_save = []
            for i, seq in enumerate(m.stateseqs):
                if i % 2 == 1:
                    continue
                temp_save.append(np.zeros((len(seq), 4)))
                sess_states = np.unique(seq)
                for s in sess_states:
                    temp_save[-1][seq == s] = m.obs_distns[s].weights[i]
                temp_save[-1] = temp_save[-1].reshape(len(seq) * 4)
            glm_weights.append(np.concatenate(temp_save))

        return np.array(glm_weights)


def find_good_chains(chains, reduce_to=8):
    delete_n = chains.shape[0] // 2 - reduce_to
    mins = np.zeros(1 + delete_n)
    n_chains = chains.shape[0] // 2
    print("Without removals: {}".format(r_hat_array_comp(chains)))
    r_hat, _ = r_hat_array_comp(chains)
    mins[0] = r_hat

    def f1(x): return 2 * x
    def f2(x): return 2 * x + 1

    for i in range(delete_n):
        print()
        r_hat_min = 10
        sol = 0
        for x in combinations(range(n_chains), i + 1):
            r_hat, _ = r_hat_array_comp(np.delete(chains, [f(y) for y in x for f in (f1, f2)], axis=0))
            if r_hat < r_hat_min:
                sol = x
            r_hat_min = min(r_hat, r_hat_min)
        print("Minimum is {} (removed {})".format(r_hat_min, i + 1))
        print("Removed: {}".format(sol))
        mins[i + 1] = r_hat_min

    return mins


class fake_result:
    def __init__(self, n):
        self.n_samples = 100


def return_ascending():
    # test function to see whether unsplit ascending sequence gives not just good r_hat, but also good ESS?
    return np.arange(100)


def return_ascending_shuffled():
    # test function to see whether shuffled ascending sequence gives better ESS
    temp = np.arange(100)
    np.random.shuffle(temp)
    return temp

def augment_weights(w):
    # we want to keep track of how much the the span of the pmf changes
    augmented_weights = np.zeros(5)
    augmented_weights[:4] = w
    pmf = weights_to_pmf(w)
    augmented_weights[-1] = max(pmf[-2:]) - min(pmf[:2])
    return augmented_weights


def compare_pmfs(test, states2compare, states_by_session, all_pmfs, title=""):
    """
       Take a set of states, and plot out their PMFs on all sessions on which they occur.
       See how different they really are.

       Takes states_by_session and all_pmfs as input from state_development
    """
    colors = ['blue', 'orange', 'green', 'black', 'red']
    assert len(states2compare) <= len(colors)
    # subtract 1 to get internal numbering
    states2compare = [s - 1 for s in states2compare]
    # transform desired states into the actual numbering, before ordering by bias
    # states2compare = [key for key in test.state_mapping.keys() if test.state_mapping[key] in states2compare]

    sessions = np.where(states_by_session[states2compare].sum(0))[0]

    for i, state in enumerate(states2compare):
        counter = 0
        for j, session in enumerate(sessions):
            plt.subplot(1, len(sessions), j + 1)
            if i == 0:
                plt.title(session)
            if states_by_session[state, session] > 0:
                plt.plot(np.where(all_pmfs[state][0])[0], (all_pmfs[state][1][counter])[all_pmfs[state][0]], c=colors[i])
                counter += 1
            plt.ylim(0, 1)
            if j != 0:
                plt.gca().set_yticks([])
    # plt.tight_layout()
    if title != "":
        plt.savefig(title)
    plt.show()

def sudden_state_changes(test, state_sets, consistencies, pmf_weights, pmfs):
    """
    Document the changes upon regression weights caused by sudden state changes.
    Also find out how weights change when first type change occurs
    Also collect info about when in session and when in training states get introduced (split by types)
    """
    n = test.results[0].n_sessions
    trial_counter = 0
    state_pmfs = {}
    state_counter = {}
    changes = [[], [], []]
    aug_changes = [[], [], []]  # also save any augmentations, such as the span of the PMF
    changes_across_types = [[], []]  # also specifically capture changes when type changes
    aug_changes_across_types = [[], []]  # also specifically capture changes when type changes
    session_time_at_sudden_change = [[], []]  # save when in a session a new state appears which brings about a new stage
    type_1_occured, type_2_occured, type_3_occured = False, False, False  # flags for keeping track

    in_sess_appear_dist = np.zeros((3, in_sess_appear_n_bins))
    in_train_appear_dist = np.zeros((3, in_train_appear_n_bins))
    in_sess_appear_dist_old = np.zeros((3, in_sess_appear_n_bins))
    in_train_appear_dist_old = np.zeros((3, in_train_appear_n_bins))

    type_1_to_2 = [[], []]  # save all previous PMFs before a sudden transition into type 2, and the first type 2 PMFs

    for seq_num in range(n):
        state_occurences = np.zeros((len(state_sets), len(test.results[0].models[0].stateseqs[seq_num])))

        for state, trials in enumerate(state_sets):
            relevant_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(test.results[0].models[0].stateseqs[seq_num]))]
            active_trials = np.zeros(len(test.results[0].models[0].stateseqs[seq_num]))

            # this has something to do with checking for every trial in the current session, how connected it is to all other trials within this state
            active_trials[relevant_trials - trial_counter] = np.sum(consistencies[tuple(np.meshgrid(relevant_trials, trials))], axis=0)
            active_trials[relevant_trials - trial_counter] -= 1
            active_trials[relevant_trials - trial_counter] = active_trials[relevant_trials - trial_counter] / (trials.shape[0] - 1)

            state_occurences[state] = active_trials

        trial_counter += len(test.results[0].models[0].stateseqs[seq_num])

        occuring_states, first_trials = [], []
        for state in range(len(state_sets)):
            occurence_trials = np.where(state_occurences[state] > 0.05)[0]
            if occurence_trials.shape[0] == 0:
                continue

            occuring_states.append(state)
            first_trials.append(occurence_trials[0])

        occuring_states = [x for _, x in sorted(zip(first_trials, occuring_states))]
        first_trials = sorted(first_trials)

        for state, first_trial in zip(occuring_states, first_trials):
            if state in state_pmfs:
                state_counter[state] += 1
                state_pmfs[state] = pmf_weights[state][state_counter[state]]
                if pmf_type(weights_to_pmf(pmf_weights[state][state_counter[state]])) == 1:
                    type_2_occured = True
                elif pmf_type(weights_to_pmf(pmf_weights[state][state_counter[state]])) == 2:
                    type_3_occured = True
            else:
                if len(state_pmfs) > 0:
                    # track appearance times
                    in_train_appear_dist[pmf_type(weights_to_pmf(pmf_weights[state][0])), int(seq_num // (n / in_train_appear_n_bins))] += 1
                    in_sess_appear_dist[pmf_type(weights_to_pmf(pmf_weights[state][0])), int(first_trial // (len(test.results[0].models[0].stateseqs[seq_num]) / in_sess_appear_n_bins))] += 1

                    # find out which of previous states are closest to new state
                    defined_points = np.zeros(11, dtype=bool)
                    defined_points[test.results[0].session_contrasts[seq_num]] = True
                    defined_points = np.logical_and(defined_points, pmfs[state][0])
                    diff = 1000  # start at a high diff to accept first value in comparison later
                    contender = np.zeros(5)

                    for existing_state in state_pmfs:
                        temp_defined_points = np.logical_and(defined_points, pmfs[existing_state][0])
                        temp_defined_points[[1, -2]] = True
                        temp_diff = np.sum(np.abs(weights_to_pmf(pmf_weights[state][0])[temp_defined_points] - weights_to_pmf(state_pmfs[existing_state])[temp_defined_points]))
                        if temp_diff < diff:
                            diff = temp_diff
                            contender = augment_weights(state_pmfs[existing_state])
                        # temp_diffs = np.abs(augment_weights(pmf_weights[state][0]) - augment_weights(state_pmfs[existing_state]))
                        # mask = temp_diffs < diffs
                        # contender[mask] = augment_weights(state_pmfs[existing_state])[mask]
                        # diffs[mask] = np.abs(augment_weights(pmf_weights[state][0]) - augment_weights(state_pmfs[existing_state]))[mask]

                    changes[pmf_type(weights_to_pmf(pmf_weights[state][0]))].append((contender[:-1], augment_weights(pmf_weights[state][0])[:-1]))
                    aug_changes[pmf_type(weights_to_pmf(pmf_weights[state][0]))].append((contender[-1], augment_weights(pmf_weights[state][0])[-1]))
                    if pmf_type(weights_to_pmf(pmf_weights[state][0])) == 0 and not type_2_occured and not type_3_occured:
                        type_1_occured = True
                    elif pmf_type(weights_to_pmf(pmf_weights[state][0])) == 1 and type_1_occured and not type_2_occured and not type_3_occured:
                        session_time_at_sudden_change[0].append(first_trial / len(test.results[0].models[0].stateseqs[seq_num]))
                        changes_across_types[0].append((contender[:-1], augment_weights(pmf_weights[state][0])[:-1]))
                        aug_changes_across_types[0].append((contender[-1], augment_weights(pmf_weights[state][0])[-1]))

                        print("Type 1 to 2")
                        print(len(state_sets) - test.state_mapping[state])
                        for existing_state in state_pmfs:
                            type_1_to_2[0].append(weights_to_pmf(state_pmfs[existing_state]))
                        type_1_to_2[1].append(weights_to_pmf(pmf_weights[state][0]))
                        type_2_occured = True
                    elif pmf_type(weights_to_pmf(pmf_weights[state][0])) == 2 and not type_3_occured and (type_2_occured or type_1_occured):
                        session_time_at_sudden_change[1].append(first_trial / len(test.results[0].models[0].stateseqs[seq_num]))
                        changes_across_types[1].append((contender[:-1], augment_weights(pmf_weights[state][0])[:-1]))
                        aug_changes_across_types[1].append((contender[-1], augment_weights(pmf_weights[state][0])[-1]))
                        type_3_occured = True
                
                in_train_appear_dist_old[pmf_type(weights_to_pmf(pmf_weights[state][0])), int(seq_num // (n / in_train_appear_n_bins))] += 1
                in_sess_appear_dist_old[pmf_type(weights_to_pmf(pmf_weights[state][0])), int(first_trial // (len(test.results[0].models[0].stateseqs[seq_num]) / in_sess_appear_n_bins))] += 1
 
                state_counter[state] = 0
                state_pmfs[state] = pmf_weights[state][0]
                if pmf_type(weights_to_pmf(pmf_weights[state][0])) == 0:
                    type_1_occured = True
                elif pmf_type(weights_to_pmf(pmf_weights[state][0])) == 1:
                    type_2_occured = True
                elif pmf_type(weights_to_pmf(pmf_weights[state][0])) == 2:
                    type_3_occured

    assert len(changes_across_types[0]) <= 1 and len(changes_across_types[1]) <= 1

    return changes, changes_across_types, aug_changes, aug_changes_across_types, in_sess_appear_dist, in_train_appear_dist, in_sess_appear_dist_old, in_train_appear_dist_old, type_1_to_2, session_time_at_sudden_change

def extrapolate_belief_strength(test, consistencies):
    n = test.results[0].n_sessions
    trial_counter = 0

    belief_sums = []
    for seq_num in range(n):

        state_consistencies = {}  # save when which state is how certain, to later compare their responses
        for state, trials in enumerate(state_sets):

            relevant_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(test.results[0].models[0].stateseqs[seq_num]))]
            active_trials = np.zeros(len(test.results[0].models[0].stateseqs[seq_num]))


            active_trials[relevant_trials - trial_counter] = np.sum(consistencies[tuple(np.meshgrid(relevant_trials, trials))], axis=0)
            active_trials[relevant_trials - trial_counter] -= 1
            active_trials[relevant_trials - trial_counter] = active_trials[relevant_trials - trial_counter] / (trials.shape[0] - 1)


            if active_trials.sum() > 0:
                state_consistencies[state] = active_trials

        trial_counter += len(test.results[0].models[0].stateseqs[seq_num])

        b_sum = np.zeros(100)
        for key in state_consistencies:
            b_sum += np.interp(np.linspace(0, len(test.results[0].models[0].stateseqs[seq_num]), 100), np.arange(len(test.results[0].models[0].stateseqs[seq_num])), state_consistencies[key])
        belief_sums.append(b_sum)

    return belief_sums


def contrasts_plot(test, state_sets, subject, save=False, show=False, dpi='figure', save_append='', consistencies=None, CMF=False):
    n = test.results[0].n_sessions
    trial_counter = 0
    cnas = [] # contrasts aNd actions

    lookback = 14
    session_pers = np.zeros((n, lookback + 1))
    total_pers = np.zeros(lookback + 1)
    total = 0

    for seq_num in range(n):
        if seq_num + 1 != 12:
            trial_counter += len(test.results[0].models[0].stateseqs[seq_num])
            continue
        c_n_a = test.results[0].data[seq_num]

        plt.figure(figsize=(18, 9))
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu']
        np.random.seed(8)
        np.random.shuffle(cmaps)
        state_consistencies = {}  # save when which state is how certain, to later compare their responses
        for state, trials in enumerate(state_sets):
            if state < len(cmaps):
                cmap = matplotlib.cm.get_cmap(cmaps[state])
            else:
                cmap = matplotlib.cm.get_cmap('Greys')
            relevant_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(test.results[0].models[0].stateseqs[seq_num]))]
            active_trials = np.zeros(len(test.results[0].models[0].stateseqs[seq_num]))

            if consistencies is None:
                active_trials[relevant_trials - trial_counter] = 1
            else:
                active_trials[relevant_trials - trial_counter] = np.sum(consistencies[tuple(np.meshgrid(relevant_trials, trials))], axis=0)
                active_trials[relevant_trials - trial_counter] -= 1
                active_trials[relevant_trials - trial_counter] = active_trials[relevant_trials - trial_counter] / (trials.shape[0] - 1)
                # print("fix this by taking the whole array, multiply by n, subtract n, divide by n-1")
                # input()

            label = "State {}".format(len(state_sets) - test.state_mapping[state]) if np.sum(relevant_trials) > 0.02 * len(test.results[0].models[0].stateseqs[seq_num]) else None

            plt.plot(active_trials, color=cmap(0.2 + 0.8 * seq_num / test.results[0].n_sessions), lw=4, label=label, alpha=0.7)
            if active_trials.sum() > 0:
                state_consistencies[len(state_sets) - test.state_mapping[state]] = active_trials

        trial_counter += len(test.results[0].models[0].stateseqs[seq_num])

        ms = 6

        cnas.append(c_n_a)

        mask = c_n_a[:, -1] == 1
        if mask.sum() > 0:
            plt.plot(np.where(mask)[0], 0.5 + 0.25 * (all_conts[cont_mapping(c_n_a[mask, 0] - c_n_a[mask, 1])]), 'o', c='b', ms=ms, label='Leftward', alpha=0.6)

        mask = c_n_a[:, -1] == 0
        if mask.sum() > 0:
            plt.plot(np.where(mask)[0], 0.5 + 0.25 * (all_conts[cont_mapping(c_n_a[mask, 0] - c_n_a[mask, 1])]), 'o', c='r', ms=ms, label='Rightward', alpha=0.6)


        for trial in range(250, 450):
            if state_consistencies[4][trial] > 0.5 and all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])] > 0 and c_n_a[trial, -1] == 1:
                plt.gca().annotate("", xytext=(trial - 2, 0.468 + 0.25 * (all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])])), xy=(trial - 0.25, 0.493 + 0.25 * (all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])])), xycoords='data', arrowprops=dict(facecolor='black', headwidth=7.5, headlength=7.5, width=2))
            if state_consistencies[6][trial] > 0.5 and all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])] < 0 and c_n_a[trial, -1] == 0:
                plt.gca().annotate("", xytext=(trial - 2, 0.468 + 0.25 * (all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])])), xy=(trial - 0.25, 0.493 + 0.25 * (all_conts[cont_mapping(c_n_a[trial, 0] - c_n_a[trial, 1])])), xycoords='data', arrowprops=dict(facecolor='black', headwidth=7.5, headlength=7.5, width=2))


        plt.title("session #{} / {}".format(1+seq_num, test.results[0].n_sessions), size=36)
        # plt.yticks(*self.cont_ticks, size=22-2)
        plt.xticks(size=fs-1)
        plt.yticks(size=fs)
        plt.ylabel('Assigned state', size=36)

        def cont_to_belief(x):
            return x * 4 - 2

        def belief_to_cont(x):
            return x / 4 + 2

        secax_y2 = plt.gca().secondary_yaxis(
            1.03, functions=(cont_to_belief, belief_to_cont))
        secax_y2.set_ylabel("Contrast", size=36)
        secax_y2.set_yticks([-1, -0.5, 0., 0.5, 1])
        secax_y2.spines['right'].set_bounds(-1, 1)
        secax_y2.tick_params(axis='y', which='major', labelsize=fs)
        plt.xlabel('Trial', size=36)
        sns.despine()

        plt.xlim(left=250, right=450)
        if test.results[0].name == 'KS014':
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1, 0, 2, 3, 4]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, fontsize=22, ncol=2, loc=(0.7, 0.05))
        else:
            plt.legend(frameon=False, fontsize=22, ncol=2, loc=(0.7, 0.05))
        plt.tight_layout()
        if save:
            print("saving with {} dpi".format(dpi))
            plt.savefig("dynamic_GLM_figures/all posterior and contrasts {}, sess {}{}.png".format(subject, seq_num, save_append), dpi=dpi)#, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        if c_n_a.shape[0] > 50:
            answers = c_n_a[lookback:, -1]
            prev_answers = c_n_a[:, -1]
            lefts = np.where(answers == 0)[0]
            rights = np.where(answers == 1)[0]
            for l in lefts:
                session_pers[seq_num] += (prev_answers[l: l + lookback + 1] == 0) * 2 - 1
            for r in rights:
                session_pers[seq_num] += (prev_answers[r: r + lookback + 1] == 1) * 2 - 1

            total_pers += session_pers[seq_num]
            session_pers[seq_num] = session_pers[seq_num] / answers.shape[0]
            total += answers.shape[0]


        if CMF and not test.results[0].name.startswith('Sim_'):
            rt_data = pickle.load(open("./session_data/{} rt info {}".format(subject, seq_num + 1), 'rb'))
            rt_data = rt_data[1:]
            assert c_n_a.shape[0] == rt_data.shape[0]
            df = pd.DataFrame(rt_data, columns = ['Signed contrast', 'RT', 'Responses'])
            means = df.groupby('Signed contrast').mean()['RT']
            stds = df.groupby('Signed contrast').sem()['RT']
            plt.errorbar(means.index, means.values, stds.values)
            plt.title(seq_num + 1, size=22)
            plt.show()

    return total_pers / total, cnas, state_consistencies

def pmf_regressions(states_by_session, pmfs, durs):
    # find out whether pmfs regressed
    # return: [total # of regressions, # of sessions, # of regressions during type 1, type 2, type 3], the reward differences
    state_perfs = {}
    state_counter = {}
    current_best_state = -1
    counter = 0
    types = [0, 0, 0]
    diffs = []
    regressed_or_not = np.zeros(states_by_session.shape[1] - 1)  # save whether sessions were regressed
    regression_magnitude = np.zeros(states_by_session.shape[1] - 1)  # how big was the regression

    for sess in range(states_by_session.shape[1]):
        states = np.where(states_by_session[:, sess])[0]

        just_introduced = []
        for s in states:
            if s not in state_counter:
                just_introduced.append(s)
                state_counter[s] = -1
            state_counter[s] += 1
            state_perfs[s] = pmf_to_perf(pmfs[s][1][state_counter[s]])
            if current_best_state == -1 or state_perfs[current_best_state] < state_perfs[s]:
                current_best_state = s

        if state_perfs[np.argmax(states_by_session[:, sess])] + 0.025 < state_perfs[current_best_state] and current_best_state not in just_introduced:
            counter += 1
            if sess < durs[0]:
                types[0] += 1
            elif sess < durs[0] + durs[1]:
                types[1] += 1
            else:
                types[2] += 1
            a, b = state_perfs[np.argmax(states_by_session[:, sess])], state_perfs[current_best_state]
            print("Regression in session {} during {:.2f}% of session ({:.2f} instead of {:.2f})".format(sess + 1, np.max(states_by_session[:, sess]) * 100, a, b))
            diffs.append(b - a)
            regressed_or_not[sess - 1] += 1
            regression_magnitude[sess - 1] = b - a

    return [counter, states_by_session.shape[1], *types], diffs, regressed_or_not, regression_magnitude


def control_flow(test, indices, trials, func_init, first_for, second_for, end_first_for):
    # Generalised control flow for iterating over samples of mode across sessions
    trial_counter = 0
    results = func_init()

    for j, only_for_length in enumerate(test.results[0].models[0].stateseqs):
        session_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(only_for_length))]
        if session_trials.shape[0] == 0:
            trial_counter += len(only_for_length)
            continue
        first_for(test, results)

        counter = -1
        for i, m in enumerate([item for sublist in test.results for item in sublist.models]):
            if i not in indices:
                continue
            counter += 1
            second_for(m, j, counter, session_trials, trial_counter, results)

        end_first_for(results, indices, j, trial_counter=trial_counter, session_trials=session_trials)
        trial_counter += len(only_for_length)

    return results


def state_pmfs(test, trials, indices):
    # Find out what the PMFs were, on which sessions they were, what their weights were, how many trials they had
    # ! thanks to the trials list, this only looks at trials assigned to the current state
    def func_init(): return {'pmfs': [], 'session_js': [], 'pmf_weights': [], 'trial_ns': []}

    def first_for(test, results):
        results['pmf'] = np.zeros(test.results[0].n_contrasts)
        results['pmf_weight'] = np.zeros(4)

    def second_for(m, j, counter, session_trials, trial_counter, results):
        # find the states (in this sample) which are assigned to the trials of this state (as defined post-hoc), and their number
        states, counts = np.unique(m.stateseqs[j][session_trials - trial_counter], return_counts=True)
        for sub_state, c in zip(states, counts):
            results['pmf'] += weights_to_pmf(m.obs_distns[sub_state].weights[j]) * c / session_trials.shape[0]
            results['pmf_weight'] += m.obs_distns[sub_state].weights[j] * c / session_trials.shape[0]

    def end_first_for(results, indices, j, **kwargs):
        results['pmfs'].append(results['pmf'] / len(indices))
        results['pmf_weights'].append(results['pmf_weight'] / len(indices))
        results['session_js'].append(j)
        results['trial_ns'].append(kwargs['session_trials'].shape[0])

    results = control_flow(test, indices, trials, func_init, first_for, second_for, end_first_for)
    return results['session_js'], results['pmfs'], results['pmf_weights'], results['trial_ns']


def state_weights(test, trials, indices):
    def func_init(): return {'weightss': [], 'session_js': []}

    def first_for(test, results):
        results['weights'] = np.zeros(test.results[0].models[0].obs_distns[0].weights.shape[1])

    def second_for(m, j, counter, session_trials, trial_counter, results):
        states, counts = np.unique(m.stateseqs[j][session_trials - trial_counter], return_counts=True)
        for sub_state, c in zip(states, counts):
            results['weights'] += m.obs_distns[sub_state].weights[j] * c / session_trials.shape[0]

    def end_first_for(results, indices, j, **kwargs):
        results['weightss'].append(results['weights'] / len(indices))
        results['session_js'].append(j)

    results = control_flow(test, indices, trials, func_init, first_for, second_for, end_first_for)
    return results['session_js'], results['weightss']

def state_development(test, state_sets, indices, save=True, save_append='', show=True, dpi='figure', separate_pmf=False, type_coloring=True, dont_plot=[], plot_until=100):
    # Now also returns durs of state types and state type summary array
    state_sets = [np.array(s) for s in state_sets]

    if test.results[0].name.startswith('GLM_Sim_'):
        print("./glm sim mice/truth_{}.p".format(test.results[0].name))
        truth = pickle.load(open("./glm sim mice/truth_{}.p".format(test.results[0].name), "rb"))
        if test.results[0].name == "GLM_Sim_18":
          truth['state_map'][3] = 4
          truth['state_map'][4] = 3
        # truth['state_posterior'] = truth['state_posterior'][:, [0, 1, 3, 4, 5, 6, 7]]  # 17, mode 2
        # truth['weights'] = [w for i, w in enumerate(truth['weights']) if i != 2]  # 17, mode 2

    states_by_session = np.zeros((len(state_sets), test.results[0].n_sessions))
    trial_counter = 0
    for i, length in enumerate([len(s) for s in test.results[0].models[0].stateseqs]):
        for state, trials in enumerate(state_sets):
            states_by_session[state, i] += np.sum(np.logical_and(trial_counter <= trials, trials < trial_counter + length)) / length
        trial_counter += length
    fig = plt.figure(figsize=(16, 9))
    spec = gridspec.GridSpec(ncols=100, nrows=3 * len(state_sets)+5, figure=fig)
    spec.update(hspace=0.)  # set the spacing between axes.
    ax0 = fig.add_subplot(spec[:5, :79])  # performance line
    ax1 = fig.add_subplot(spec[6:, :79])  # state lines
    ax2 = fig.add_subplot(spec[6:, 88:])

    ax0.plot(1 + 0.25, 0.6, 'ko', ms=18)
    ax0.plot(1 + 0.25, 0.6, 'wo', ms=16.8)
    ax0.plot(1 + 0.25, 0.6, 'ko', ms=16.8, alpha=abs(num_to_cont[0]))

    ax0.plot(1 + 0.25, 0.6 + 0.2, 'ko', ms=18)
    ax0.plot(1 + 0.25, 0.6 + 0.2, 'wo', ms=16.8)
    ax0.plot(1 + 0.25, 0.6 + 0.2, 'ko', ms=16.8, alpha=abs(num_to_cont[1]))

    if test.results[0].type != 'bias' and not test.results[0].name.startswith('fip_'):
        current, counter = 0, 0
        for c in [2, 3, 4, 5]:
            if test.results[0].infos[c] == current:
                counter += 1
            else:
                counter = 0
                ax0.axvline(test.results[0].infos[c] + 1, color='gray', zorder=0)
                ax1.axvline(test.results[0].infos[c] + 1, color='gray', zorder=0)
                ax0.plot(test.results[0].infos[c] + 1 - 0.25, 0.6 + counter * 0.2, 'ko', ms=18)
                ax0.plot(test.results[0].infos[c] + 1 - 0.25, 0.6 + counter * 0.2, 'wo', ms=16.8)
                ax0.plot(test.results[0].infos[c] + 1 - 0.25, 0.6 + counter * 0.2, 'ko', ms=16.8, alpha=abs(num_to_cont[c]))
            current = test.results[0].infos[c]
    # if test.results[0].type == 'all' or test.results[0].type == 'prebias_plus':
    #     ax1.axvline(test.results[0].infos['bias_start'] + 1 - 0.5, color='gray', zorder=0)
    #     ax0.axvline(test.results[0].infos['bias_start'] + 1 - 0.5, color='gray', zorder=0)
    #     ax0.annotate('Bias', (test.results[0].infos['bias_start'] + 1 - 0.5, 0.68), fontsize=22)

    all_pmfs = []
    all_pmf_weights = []
    all_trial_ns = []
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu']
    np.random.seed(8)
    np.random.shuffle(cmaps)

    # find out state ordering
    pmfs_to_score = []
    for state, trials in enumerate(state_sets):
        if separate_pmf:
            n_trials = len(trials)
            session_js, pmfs, _, _ = state_pmfs(test, trials, indices)
        else:
            pmfs = np.zeros((len(indices), test.results[0].n_contrasts))
            n_trials = len(trials)
            counter = 0
            for i, m in enumerate([item for sublist in test.results for item in sublist.models]):
                if i not in indices:
                    continue
                trial_counter = 0
                for j, state_seq in enumerate(m.stateseqs):
                    session_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(state_seq))]
                    if session_trials.shape[0] > 0:
                        states, counts = np.unique(state_seq[session_trials - trial_counter], return_counts=True)
                        for sub_state, c in zip(states, counts):
                            pmfs[counter] += weights_to_pmf(m.obs_distns[sub_state].weights[j]) * c / n_trials
                    trial_counter += len(state_seq)
                counter += 1
        pmfs_to_score.append(np.mean(pmfs))
    # test.state_mapping = dict(zip(range(len(state_sets)), np.argsort(np.argsort(pmfs_to_score))))  # double argsort for ranks
    test.state_mapping = dict(zip(np.flip(np.argsort((states_by_session != 0).argmax(axis=1))), range(len(state_sets))))
    print(test.state_mapping)
    if subject == "KS014":
        test.state_mapping[4] = 5
        test.state_mapping[1] = 4
        test.state_mapping[3] = 2
        test.state_mapping[0] = 1

    for state, trials in enumerate(state_sets):
        cmap = matplotlib.cm.get_cmap(cmaps[state]) if state < len(cmaps) else matplotlib.cm.get_cmap('Greys')

        if separate_pmf:
            n_trials = len(trials)
            session_js, pmfs, pmf_weights, trial_ns = state_pmfs(test, trials, indices)
        else:
            pmfs = np.zeros((len(indices), test.results[0].n_contrasts))
            pmf_weights = np.zeros((len(indices), test.results[0].obs_distns[0].weights.shape[0]))
            n_trials = len(trials)
            counter = 0
            for i, m in enumerate([item for sublist in test.results for item in sublist.models]):
                if i not in indices:
                    continue
                trial_counter = 0
                for j, state_seq in enumerate(m.stateseqs):
                    session_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(state_seq))]
                    if session_trials.shape[0] > 0:
                        states, counts = np.unique(state_seq[session_trials - trial_counter], return_counts=True)
                        for sub_state, c in zip(states, counts):
                            pmfs[counter] += weights_to_pmf(m.obs_distns[sub_state].weights[j]) * c / n_trials
                            pmf_weights[counter] += m.obs_distns[sub_state].weights[j] * c / n_trials
                    trial_counter += len(state_seq)
                counter += 1

        trial_counter = 0
        session_max = 0
        for j, state_seq in enumerate(test.results[0].models[0].stateseqs):
            session_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(state_seq))]
            if session_trials.shape[0] > 0:
                session_max = j
            trial_counter += len(state_seq)

        defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
        if not separate_pmf:
            temp = np.sum(pmfs[:, defined_points]) / (np.sum(defined_points))
            state_color = colors[int(temp * 101 - 1)]

            if not test.state_mapping[state] in dont_plot:
                ax1.fill_between(range(1, 1 + test.results[0].n_sessions), test.state_mapping[state] - 0.5,
                                 test.state_mapping[state] + states_by_session[state] - 0.5, color=state_color)

        else:
            n_points = 400
            points = np.linspace(1, test.results[0].n_sessions, n_points)
            interpolation = np.interp(points, np.arange(1, 1 + test.results[0].n_sessions), states_by_session[state])

            for k in range(n_points-1):
                if not test.state_mapping[state] in dont_plot:
                    ax1.fill_between([points[k], points[k+1]],
                                     test.state_mapping[state] - 0.5, [test.state_mapping[state] + interpolation[k] - 0.5, test.state_mapping[state] + interpolation[k+1] - 0.5], color=cmap(0.3 + 0.7 * k / n_points))
        ax1.annotate(len(state_sets) - test.state_mapping[state], (test.results[0].n_sessions + 0.05, test.state_mapping[state] - 0.15), fontsize=22, annotation_clip=False)

        alpha_level = 0.3
        ax2.axvline(0.5, c='grey', alpha=alpha_level, zorder=4)

        if test.results[0].type == 'bias':
            defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
        # else:
        #     if self.active_during_bias[self.state_map[s]]:
        #         defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
        #     else:
        #         defined_points = np.zeros(test.results[0].n_contrasts, dtype=bool)
        #         defined_points[[0, 1, -2, -1]] = True
        if separate_pmf:
            for j, pmf, pmf_weight, trial_n in zip(session_js, pmfs, pmf_weights, trial_ns):
                if not test.state_mapping[state] in dont_plot:
                    ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), pmf[defined_points] - 0.5 + test.state_mapping[state], color=cmap(0.2 + 0.8 * j / test.results[0].n_sessions))
                    ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), pmf[defined_points] - 0.5 + test.state_mapping[state], ls='', ms=7, marker='*', color=cmap(j / test.results[0].n_sessions))
            all_pmfs.append((defined_points, pmfs))
            all_pmf_weights.append(pmf_weights)
            all_trial_ns.append(trial_ns)
        else:
            temp = np.percentile(pmfs, [2.5, 97.5], axis=0)
            if not test.state_mapping[state] in dont_plot:
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), pmfs[:, defined_points].mean(axis=0) - 0.5 + test.state_mapping[state], color=state_color)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), pmfs[:, defined_points].mean(axis=0) - 0.5 + test.state_mapping[state], ls='', ms=7, marker='*', color=state_color)
                ax2.fill_between(np.where(defined_points)[0] / (len(defined_points)-1), temp[1, defined_points] - 0.5 + test.state_mapping[state], temp[0, defined_points] - 0.5 + test.state_mapping[state], alpha=0.2, color=state_color)
            all_pmfs.append((defined_points, pmfs[:, defined_points].mean(axis=0)))

        if not test.state_mapping[state] in dont_plot:
            ax2.annotate("Type {}".format(1 + pmf_type(pmf[defined_points])), (1.05, test.state_mapping[state] - 0.37), rotation=90, size=13, color=type2color[pmf_type(pmf)], annotation_clip=False)
        ax2.axhline(test.state_mapping[state] + 0.5, c='k')
        ax2.axhline(test.state_mapping[state], c='grey', alpha=alpha_level, zorder=4)
        ax1.axhline(test.state_mapping[state] + 0.5, c='grey', alpha=alpha_level, zorder=4)

    if test.results[0].name.startswith('GLM_Sim_'):
        for state in range(min(len(state_sets), len(truth['weights']))):
            ax1.plot(range(1, 1 + test.results[0].n_sessions), len(state_sets) - state + truth['state_posterior'][:, truth['state_map'][state]] - 1.5, color='r', lw=2)
            ax1.plot(range(1, 1 + test.results[0].n_sessions), len(state_sets) - state + truth['state_posterior'][:, truth['state_map'][state]] - 1.5, color='k', linestyle='--', lw=2)
            if test.results[0].name.startswith('GLM_Sim_11'):
                sim_pmf = weights_to_pmf(truth['weights'][0])
                defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='r', lw=2)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='k', linestyle='--', lw=2)
                sim_pmf = weights_to_pmf(truth['weights'][-1])
                defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='r', lw=2)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='k', linestyle='--', lw=2)
            else:
                sim_pmf = weights_to_pmf(truth['weights'][truth['state_map'][state]])
                defined_points = np.ones(test.results[0].n_contrasts, dtype=bool)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='r', lw=2)
                ax2.plot(np.where(defined_points)[0] / (len(defined_points)-1), sim_pmf[defined_points] - 1.5 + len(state_sets) - state, color='k', linestyle='--', lw=2)

    ax1.annotate("State #", (test.results[0].n_sessions - 0.05, len(state_sets) - 0.45), fontsize=14, annotation_clip=False)
    if not test.results[0].name.startswith('Sim_'):
        perf = np.zeros(test.results[0].n_sessions)
        found_files = 0
        counter = test.results[0].infos['bias_start'] - 1 if test.results[0].type == 'bias' else -1
        while found_files < test.results[0].n_sessions:
            counter += 1
            if counter > 1000:
                break
            try:
                feedback = pickle.load(open(file_prefix + "/session_data/{}_side_info_{}.p".format(test.results[0].name, counter), "rb"))
            except FileNotFoundError:
                continue
            perf[found_files] = np.mean(feedback[:, 1])
            found_files += 1
        ax0.axhline(-0.5, c='k')
        ax0.axhline(0.5, c='k')
        print(perf)
        ax0.fill_between(range(1, 1 + test.results[0].n_sessions), perf - 0.5, -0.5, color='k')

    if not test.results[0].name.startswith('GLM_Sim_') and not test.results[0].name.startswith('fip_'):
        durs, state_types = state_type_durs(states_by_session, all_pmfs)

        # how many states per session per type
        states_per_sess = np.sum(states_by_session > 0.05, axis=0)
        if durs[0] > 0 and durs[1] > 0 and durs[2] > 1:
            states_per_type = [np.mean(states_per_sess[:durs[0]]), np.mean(states_per_sess[durs[0]:durs[0]+durs[1]]), np.mean(states_per_sess[durs[0]+durs[1]:])]
        else:
            states_per_type = []
        # other statistics
        dur_counter = 1
        contrast_intro_types = [0, 0, 0, 0]
        state, when = np.where(states_by_session > 0.05)
        introductions_by_stage = np.zeros(3)
        covered_states = []
        for i, d in enumerate(durs):
            if type_coloring:
                ax0.fill_between(range(dur_counter, min(1 + dur_counter + d, plot_until)), 0.5, -0.5, color=type2color[i], zorder=0, alpha=0.3)
            dur_counter += d

            # find out during which state type which contrast was introduced
            for j, contrast in enumerate([2, 3, 4, 5]):
                if contrast_intro_types[j] != 0:
                    continue
                if test.results[0].infos[contrast] + 1 < dur_counter:
                    contrast_intro_types[j] = i+1

            # find out during which stage which state was introduced
            for s in range(len(state_sets)):
                if np.sum(state == s) == 0 or s in covered_states:
                    continue
                if when[state == s][0] + 1 < dur_counter:
                    introductions_by_stage[i] += 1
                    covered_states.append(s)

    ax2.set_title('Psychometric\nfunction', size=16)
    ax1.set_ylabel('Proportion of trials', size=28 if len(state_sets) > 3 else 20, labelpad=-20 if len(state_sets) > 4 else 0)
    ax0.set_ylabel('% correct', size=18)
    ax2.set_ylabel('P(rightwards answer)', size=26 if len(state_sets) > 3 else 18, labelpad=-20 if len(state_sets) > 4 else 0)
    ax1.set_xlabel('Session', size=28)
    ax2.set_xlabel('Contrast', size=26)
    ax1.set_xlim(left=1, right=test.results[0].n_sessions)
    ax0.set_xlim(left=1, right=test.results[0].n_sessions)
    ax2.set_xlim(left=0, right=1)
    ax1.set_ylim(bottom=-0.5, top=len(state_sets) - 0.5)
    ax0.set_ylim(bottom=-0.5)
    ax0.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.set_ylim(bottom=-0.5, top=len(state_sets) - 0.5)
    y_pos = [item for sublist in [[i, i+0.5] for i in range(len(state_sets))] for item in sublist]
    y_pos.insert(0, -0.5)
    ax1.set_yticks(np.array(y_pos))
    ax1.set_yticklabels([0, 0.5, 1] + ['', ''] * (len(state_sets) - 1))
    if len(state_sets) > 1:
        ax0.set_yticks([-0.5, 0, 0.5])
        ax0.set_yticklabels([0, 0.5, 1], size=fs)
    else:
        ax0.set_yticks([0, 0.5])
        ax0.set_yticklabels([0.5, 1], size=fs)
    ax0.set_xticks([])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([0, 0.5, 1] + ['', ''] * (len(state_sets) - 1), size=fs)

    # red session marker
    # ax1.axvline(12, color='red', zorder=0)
    # ax0.axvline(12, color='red', zorder=0)

    ax1.tick_params(axis='both', labelsize=fs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels([-1, 0, 1], size=fs)

    plt.tight_layout()
    if save:
        print("saving with {} dpi".format(dpi))
        plt.savefig(file_prefix + "/dynamic_GLM_figures/meta_state_development_{}_{}{}.png".format(test.results[0].name, separate_pmf, save_append), dpi=dpi)
    if show:
        plt.show()
        plt.close()
    else:
        plt.close()

    if test.results[0].name.startswith("GLM_Sim_") and 'durs' in truth and False:
        plt.figure(figsize=(16, 9))
        for state, trials in enumerate(state_sets):
            if state > len(truth['weights']):
                continue
            dur_params = dur_hists(test, trials, indices)

            plt.subplot(4, 5, 1 + 2 * test.state_mapping[state])
            plt.hist(dur_params[:, 0])
            plt.axvline(truth['durs'][truth['state_map'][state]][0], color='red')

            plt.subplot(4, 5, 2 + 2 * test.state_mapping[state])
            plt.hist(dur_params[:, 1])
            plt.axvline(truth['durs'][truth['state_map'][state]][1], color='red')

        plt.tight_layout()
        plt.savefig("dur hists")
        plt.show()

        from scipy.stats import nbinom
        points = np.arange(900)
        plt.figure(figsize=(16, 9))
        for state, trials in enumerate(state_sets):
            dur_params = dur_hists(test, trials, indices)

            plt.subplot(2, 4, 1 + test.state_mapping[state])
            plt.plot(nbinom.pmf(points, np.mean(dur_params[:, 0]), np.mean(dur_params[:, 1])))
            plt.plot(nbinom.pmf(points, truth['durs'][truth['state_map'][state]][0], truth['durs'][truth['state_map'][state]][1]), color='red')
            plt.xlabel("# of trials")
            plt.ylabel("P")

        plt.tight_layout()
        plt.savefig("dur dists")
        plt.show()

    if not test.results[0].name.startswith('GLM_Sim_') and not test.results[0].name.startswith('fip_'):     
        return states_by_session, all_pmfs, all_pmf_weights, durs, state_types, contrast_intro_types, smart_divide(introductions_by_stage, np.array(durs)), introductions_by_stage, states_per_type, all_trial_ns
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def dur_hists(test, trials, indices):
    trial_counter = 0
    results = {'dur_params': np.zeros((len(indices), 2))}

    for j, only_for_length in enumerate(test.results[0].models[0].stateseqs):
        session_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(only_for_length))]
        if session_trials.shape[0] == 0:
            trial_counter += len(only_for_length)
            continue

        quit_out = False
        counter = -1
        for i, m in enumerate([item for sublist in test.results for item in sublist.models]):
            if i not in indices:
                continue
            counter += 1
            states, counts = np.unique(m.stateseqs[j][session_trials - trial_counter], return_counts=True)
            for sub_state, c in zip(states, counts):
                try:
                    p = m.dur_distns[sub_state].p_save  # this cannot be accessed after deleting data, but not every model's data is deleted
                except:
                    p = m.dur_distns[sub_state].p
                results['dur_params'][counter] += np.array([m.dur_distns[sub_state].r * c / session_trials.shape[0], p * c / session_trials.shape[0]])
                quit_out = True
            
        if quit_out:
            return results['dur_params']

        trial_counter += len(only_for_length)


def smart_divide(a, b):
    c = np.zeros_like(a)
    d = np.logical_and(a == 0, b == 0)
    c[~d] = a[~d] / b[~d]
    return c


def predictive_check(test, indices, weights=1, state_assistance=False, return_acc_plot=False, till_session=None):

    title_add = '' if not state_assistance else '_state_assistance'

    decay = 0.25
    perseveration_filter_normalization = np.exp(- decay * np.arange(5000)).sum()
    decay_const = np.exp(- decay)

    true_accs = []
    gen_accs = []
    true_pers = []
    gen_pers = []
    true_accs_total = []
    true_pers_total = []
    true_accs_50 = []
    true_pers_50 = []

    # we save a bunch of different things, such as 50 percentiles, but also performance sample minus last 50 trials, leading to bad naming schemes heres, TODO
    gen_accs_save = []  # excluding 25 first and 50 last trials
    gen_accs_save_total = []  # all trials
    gen_accs_save_50 = []  # excluding last 50 trials
    gen_accs_50 = []
    gen_accs_95 = []

    # what if we exclude the strong cont.s
    true_accs_total_weak = []
    true_accs_total_weakest = []
    gen_accs_save_total_weak = []
    gen_accs_save_total_weakest = []

    weighted_true_accs_total = []
    weighted_gen_accs_save_total = []


    gen_pers_save = []  # excluding 25 first and 50 last trials
    gen_pers_save_total = []  # all trials
    gen_pers_save_50 = []  # excluding last 50 trials
    gen_pers_50 = []
    gen_pers_95 = []

    pmf_percentiles_total = []
    pmf_percentiles_50 = []
    pmf_percentiles_75 = []

    pmf_dists = []
    acc_dists = []

    all_models = [item for sublist in test.results for item in sublist.models]

    subject_timeouts = pickle.load(open(f"./session_data/subject_timeout_{test.results[0].name}", 'rb'))

    for session in range(test.results[0].n_sessions):
        print(session, test.results[0].data[session].shape)
        datas = test.results[0].data[session]

        responses = np.zeros(datas.shape[0])
        all_resp = []
        for index in list(indices) * 3:
            model = all_models[index]
        # for model in np.random.choice(all_models, 1200):
            curr_state = model.stateseqs[session][0]

            # simulate from the modle giving the starting state
            generated_trials = 0
            perseveration = 0

            responses_randoms = np.random.rand(datas.shape[0])

            while generated_trials < datas.shape[0]:
                p, r = model.dur_distns[curr_state].p_save, model.dur_distns[curr_state].r
                state_trials = 1 + np.random.poisson(np.random.gamma(r, p / (1 - p)))  # start at 1

                # print(f"{state_trials} out of {datas.shape[0]}, pmf: {weights_to_pmf(model.obs_distns[curr_state].weights[session])[[0, 1, -2, -1]]}")

                for i in range(state_trials):
                    if generated_trials + i >= datas.shape[0]:
                        break
                    local_data = datas[generated_trials + i, :-1]
                    # replace with actual, model-generated perseveration
                    if generated_trials + i in subject_timeouts[session]:
                        # print()
                        # print(perseveration)
                        perseveration = perseveration * decay_const ** subject_timeouts[session][generated_trials + i]
                        # print(perseveration)
                    local_data[2] = perseveration / perseveration_filter_normalization

                    if state_assistance:
                        curr_state = model.stateseqs[session][generated_trials + i]
                    responses[generated_trials + i] = responses_randoms[generated_trials + i] < 1 / (1 + np.exp(- np.sum(model.obs_distns[curr_state].weights[session] * local_data)))  # compute this directly, calling rvs takes too long

                    perseveration = (responses[generated_trials + i] * 2 - 1) + perseveration * decay_const
                generated_trials += state_trials

                # get the next state
                if not state_assistance:
                    curr_state = np.random.choice(15, p=model.trans_distn.trans_matrix[curr_state])

            all_resp.append(responses.copy())

        # maybe TODO: perseveration estimation would really need to know about timeouts
        gen_accs_save_total.append([compute_accuracy(datas, resp) for resp in all_resp])
        gen_pers_save_total.append([compute_perseveration(datas, resp) for resp in all_resp])

        true_accs_total.append(compute_accuracy(datas, datas[:, -1]))
        true_pers_total.append(compute_perseveration(datas, datas[:, -1]))

        # weighted stuff
        # weighted_true_accs_total.append(compute_accuracy(datas, datas[:, -1], weights[session]))
        # weighted_gen_accs_save_total.append([compute_accuracy(datas, resp, weights[session]) for resp in all_resp])


        true_accs_total_weak.append(compute_accuracy_weak_conts(datas, datas[:, -1], exclude_list=[1.]))
        # true_accs_total_weakest.append(compute_accuracy_weak_conts(datas, datas[:, -1], exclude_list=[1., 0.987]))
        gen_accs_save_total_weak.append([compute_accuracy_weak_conts(datas, resp, exclude_list=[1.]) for resp in all_resp])
        # gen_accs_save_total_weakest.append([compute_accuracy_weak_conts(datas, resp, exclude_list=[1., 0.987]) for resp in all_resp])

        true_pmf = compute_pmf(datas, datas[:, -1])
        gen_pmf = [compute_pmf(datas, resp) for resp in all_resp]
        pmf_percentiles_total.append(calculate_true_percentiles(true_pmf, np.array(gen_pmf).T))

        pmf_dists.append((true_pmf - np.mean(gen_pmf, axis=0)) / np.std(gen_pmf, axis=0))

        if datas.shape[0] > 75:
            true_accs.append(compute_accuracy(datas[25:-50], datas[25:-50, -1]))
            true_accs_50.append(compute_accuracy(datas[:-50], datas[:-50, -1]))
            all_accs = [compute_accuracy(datas[25:-50], resp[25:-50]) for resp in all_resp]
            gen_accs.append(np.mean(all_accs))
            gen_accs_save.append([compute_accuracy(datas[25:-50], resp[25:-50]) for resp in all_resp])
            gen_accs_save_50.append([compute_accuracy(datas[:-50], resp[:-50]) for resp in all_resp])
            gen_accs_50.append([np.percentile(all_accs, 25), np.percentile(all_accs, 75)])
            gen_accs_95.append([np.percentile(all_accs, 2.5), np.percentile(all_accs, 97.5)])

            true_pers.append(compute_perseveration(datas[25:-50], datas[25:-50, -1]))
            true_pers_50.append(compute_perseveration(datas[:-50], datas[:-50, -1]))
            all_pers = [compute_perseveration(datas[25:-50], resp[25:-50]) for resp in all_resp]
            gen_pers.append(np.mean(all_pers))
            gen_pers_save.append([compute_perseveration(datas[25:-50], resp[25:-50]) for resp in all_resp])
            gen_pers_save_50.append([compute_perseveration(datas[:-50], resp[:-50]) for resp in all_resp])
            gen_pers_50.append([np.percentile(all_pers, 25), np.percentile(all_pers, 75)])
            gen_pers_95.append([np.percentile(all_pers, 2.5), np.percentile(all_pers, 97.5)])

            true_pmf = compute_pmf(datas[25:-50], datas[25:-50, -1])
            gen_pmf = [compute_pmf(datas[25:-50], resp[25:-50]) for resp in all_resp]
            pmf_percentiles_75.append(calculate_true_percentiles(true_pmf, np.array(gen_pmf).T))

            true_pmf = compute_pmf(datas[:-50], datas[:-50, -1])
            gen_pmf = [compute_pmf(datas[:-50], resp[:-50]) for resp in all_resp]
            pmf_percentiles_50.append(calculate_true_percentiles(true_pmf, np.array(gen_pmf).T))

            if session == till_session:
                break

        # plt.figure(figsize=(16, 9))
        # plt.plot(true_pmf, 'k', zorder=3, label='Empirical')
        # plt.plot(np.mean(gen_pmf, axis=0), 'g', zorder=2, label='Generated mean')
        # plt.plot(np.min(gen_pmf, axis=0), 'k--', alpha=0.25, zorder=2)
        # plt.plot(np.max(gen_pmf, axis=0), 'k--', alpha=0.25, zorder=2)
        # percentiles = np.percentile(gen_pmf, [25, 75], axis=0)
        # plt.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='b', zorder=1, alpha=0.25, label='Generated 50% CI')
        # percentiles = np.percentile(gen_pmf, [2.5, 97.5], axis=0)
        # plt.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='r', zorder=0, alpha=0.1, label='Generated 95% CI')

        # if len(true_pmf) == 4:
        #     plt.xticks(range(4), [-1, -0.5, 0.5, 1])
        # elif len(true_pmf) == 6:
        #     plt.xticks(range(6), [-1, -0.5, -0.25, 0.25, 0.5, 1])
        # elif len(true_pmf) == 8:
        #     plt.xticks(range(8), [-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1])
        # elif len(true_pmf) == 10:
        #     plt.xticks(range(10), [-1, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1])
        # elif len(true_pmf) == 11:
        #     plt.xticks(range(11), [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1])
        # elif len(true_pmf) == 9:
        #     plt.xticks(range(9), [-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])

        # plt.legend(frameon=False, fontsize=24)
        # plt.xlim(0, len(true_pmf) - 1)
        # plt.gca().spines[['right', 'top']].set_visible(False)
        # plt.gca().tick_params(axis='both', which='major', labelsize=21)
        # plt.xlabel("Contrast", size=30)
        # plt.ylabel("% rightwards responses", size=30)
        # plt.ylim(0, 1)
        # plt.title("Session {}".format(session + 1), size=30)
        # plt.savefig("pmf_check_{}_{}{}_even.png".format(test.results[0].name, session, title_add))
        # plt.close()

        # if calculate_true_percentiles(true_pmf, np.array(gen_pmf).T)[0] == 0.:
        #     print(calculate_true_percentiles(true_pmf, np.array(gen_pmf).T))
        #     return true_pmf, np.array(gen_pmf).T

    if return_acc_plot:
        return true_accs, gen_accs_save_total, true_pmf, gen_pmf

    plt.figure(figsize=(16, 9))
    plt.plot(range(1, 1 + len(true_accs)), true_accs, 'k', zorder=3, label='Empirical')
    plt.plot(range(1, 1 + len(true_accs)), gen_accs, 'g', zorder=2, label='Generated mean')
    plt.fill_between(range(1, 1 + len(true_accs)), np.array(gen_accs_50)[:, 0], np.array(gen_accs_50)[:, 1], color='b', zorder=1, alpha=0.25, label='Generated 50% CI')
    plt.fill_between(range(1, 1 + len(true_accs)), np.array(gen_accs_95)[:, 0], np.array(gen_accs_95)[:, 1], color='r', zorder=0, alpha=0.1, label='Generated 95% CI')


    plt.legend(frameon=False, fontsize=24)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().tick_params(axis='both', which='major', labelsize=21)
    plt.xlabel("Session", size=30)
    plt.ylabel("Reward rate", size=30)
    plt.ylim(0, 1)
    plt.savefig("./pred_checks/all_pred_check_{}_{}{}.png".format('acc', test.results[0].name, title_add))
    plt.close()


    plt.figure(figsize=(16, 9))
    plt.plot(range(1, 1 + len(true_pers)), true_pers, 'k', zorder=3, label='Empirical')
    plt.plot(range(1, 1 + len(true_pers)), gen_pers, 'g', zorder=2, label='Generated mean')
    plt.fill_between(range(1, 1 + len(true_accs)), np.array(gen_pers_50)[:, 0], np.array(gen_pers_50)[:, 1], color='b', zorder=1, alpha=0.25, label='Generated 50% CI')
    plt.fill_between(range(1, 1 + len(true_accs)), np.array(gen_pers_95)[:, 0], np.array(gen_pers_95)[:, 1], color='r', zorder=0, alpha=0.1, label='Generated 95% CI')

    plt.legend(frameon=False, fontsize=24)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().tick_params(axis='both', which='major', labelsize=21)
    plt.xlabel("Session", size=30)
    plt.ylabel("Response correlation", size=30)
    plt.ylim(0, 0.6)
    plt.savefig("./pred_checks/all_pred_check_{}_{}{}.png".format('pers', test.results[0].name, title_add))
    plt.close()

    acc_dists.append((np.array(true_accs_total) - np.array([np.mean(x) for x in gen_accs_save_total])) / np.array([np.std(x) for x in gen_accs_save_total]))

    # find out in which percentiles the accuracies lie
    acc_percentiles = calculate_true_percentiles(true_accs, gen_accs_save)
    pers_percentiles = calculate_true_percentiles(true_pers, gen_pers_save)

    weighted_acc_percentiles = calculate_true_percentiles(weighted_true_accs_total, weighted_gen_accs_save_total)

    acc_percentiles_50 = calculate_true_percentiles(true_accs_50, gen_accs_save_50)
    pers_percentiles_50 = calculate_true_percentiles(true_pers_50, gen_pers_save_50)

    acc_percentiles_total = calculate_true_percentiles(true_accs_total, gen_accs_save_total)
    pers_percentiles_total = calculate_true_percentiles(true_pers_total, gen_pers_save_total)

    acc_weak_percentiles = calculate_true_percentiles(true_accs_total_weak, gen_accs_save_total_weak)
    acc_weakest_percentiles = calculate_true_percentiles(true_accs_total_weakest, gen_accs_save_total_weakest)

    return acc_percentiles, pers_percentiles, pmf_percentiles_75, acc_percentiles_50, pers_percentiles_50, pmf_percentiles_50, acc_percentiles_total, \
            pers_percentiles_total, pmf_percentiles_total, acc_weak_percentiles, acc_weakest_percentiles, acc_dists, pmf_dists, weighted_acc_percentiles

def compute_accuracy(inputs, responses, weights=1):
    # inputs: n_trials x 4, responses: n_trials
    right = inputs[:, 0] > 0
    left = inputs[:, 1] > 0
    zeros = np.logical_and(inputs[:, 0] == 0, inputs[:, 1] == 0)
                           
    correct = np.zeros_like(responses)
    correct[left & (responses == 1)] = 1
    correct[right & (responses == 0)] = 1
    correct[zeros] = 0.5

    return np.mean(correct * weights)

def compute_accuracy_weak_conts(inputs, responses, exclude_list=[]):
    # inputs: n_trials x 4, responses: n_trials
    right = inputs[:, 0] > 0
    left = inputs[:, 1] > 0

    for e in exclude_list:
        right = np.logical_and(right, inputs[:, 0] != e)
        left = np.logical_and(left, inputs[:, 1] != e)

    zeros = np.logical_and(inputs[:, 0] == 0, inputs[:, 1] == 0)
                           
    correct = np.zeros_like(responses)
    correct[left & (responses == 1)] = 1
    correct[right & (responses == 0)] = 1
    correct[zeros] = 0.5

    return np.sum(correct) / (np.sum(left) + np.sum(right) + np.sum(zeros))

def compute_perseveration(inputs, responses):
    # inputs: responses: n_trials

    # sometimes all answers are the same, crashing the correlation
    if np.all(responses[:-1] == responses[0]) or np.all(responses[1:] == responses[1]):
        return 1
    
    return np.corrcoef(responses[:-1], responses[1:])[0, 1]

def compute_pmf(inputs, responses):
    conts = inputs[:, 0] - inputs[:, 1]
    perfs = np.zeros(len(np.unique(conts)))
    for i, c in enumerate(np.unique(conts)):
        perfs[i] = np.mean(responses[conts == c])

    return 1-perfs

def calculate_true_percentiles(trues, gens):
    percentiles = []
    for true, gen in zip(trues, gens):
        # Compute the percentile rank of the true accuracy within the generated accuracies array
        percentiles.append((np.sum(np.array(gen) <= np.array(true)) / len(gen) + np.sum(np.array(gen) < np.array(true)) / len(gen)) / 2)
    return percentiles

def dist_helper(dist_matrix, state_hists, inds):
    for i, j in np.nditer(inds):
        dist_matrix[i, j] = np.sum(np.abs(state_hists[i] - state_hists[j]))
    return dist_matrix

def type_2_appearance(states, pmfs):
    # How does type 2 appear, is it a new state or continuation of a type 1?
    state_counter = {}
    found_states = 0
    for session_counter in range(states.shape[1]):
        for state, pmf in zip(range(states.shape[0]), pmfs):
            if states[state, session_counter]:
                if state not in state_counter:
                    state_counter[state] = -1
                state_counter[state] += 1
                if pmf_type(pmf[1][state_counter[state]][pmf[0]]) == 1:
                    found_states += 1
                    new = state_counter[state] == 0
        if found_states > 1:
            print("Problem")
            return 2
        if found_states == 1:
            return new

def state_type_durs(states, pmfs):
    # Takes states and pmfs, first creates an array of when which type is how active, then computes the number of sessions each type lasts.
    # A type lasts until a more advanced type takes up more than 50% of a session (and cannot return)
    # Returns the durations for all the different state types, and an array which holds the state percentages
    state_types = np.zeros((4, states.shape[1]))
    for s, pmf in zip(states, pmfs):
        pmf_counter = -1
        for i in range(states.shape[1]):
            if s[i]:
                pmf_counter += 1
                state_types[pmf_type(pmf[1][pmf_counter][pmf[0]]), i] += s[i]  # indexing horror

    if np.any(state_types[1] > 0.5):
        durs = (np.where(state_types[1] > 0.5)[0][0],
                np.where(state_types[2] > 0.5)[0][0] - np.where(state_types[1] > 0.5)[0][0],
                states.shape[1] - np.where(state_types[2] > 0.5)[0][0])
        if np.where(state_types[2] > 0.5)[0][0] < np.where(state_types[1] > 0.5)[0][0]:
            durs = (np.where(state_types[2] > 0.5)[0][0],
                    0,
                    states.shape[1] - np.where(state_types[2] > 0.5)[0][0])
    else:
        durs = (np.where(state_types[2] > 0.5)[0][0],
                0,
                states.shape[1] - np.where(state_types[2] > 0.5)[0][0])
    return durs, state_types

def get_first_pmfs(states, pmfs):
    # get the first pmf of every type, also where they are defined, and whether they are the first pmf of that state
    earliest_sessions = [1000, 1000, 1000]  # high values to compare with min
    first_pmfs = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # for every type (3), we save: the pmf, the defined points, the session at which they appear
    changing_pmfs = [[0, 0], [0, 0]]  # if the new type appears first through a slow transition, we remember its defined points (pmf[1]) and its pmf trajectory pmf[1]
    for state, pmf in zip(states, pmfs):
        sessions = np.where(state)[0]
        for i, (sess_pmf, sess) in enumerate(zip(pmf[1], sessions)):
            if earliest_sessions[pmf_type(sess_pmf[pmf[0]])] > sess:
                earliest_sessions[pmf_type(sess_pmf[pmf[0]])] = sess
                first_pmfs[3 * pmf_type(sess_pmf[pmf[0]])] = sess_pmf
                first_pmfs[1 + 3 * pmf_type(sess_pmf[pmf[0]])] = pmf[0]
                first_pmfs[2 + 3 * pmf_type(sess_pmf[pmf[0]])] = i
                if i != 0:
                    changing_pmfs[pmf_type(sess_pmf[pmf[0]]) - 1] = [pmf[0], pmf[1]]
    return first_pmfs, changing_pmfs

def two_sample_binom_test(s1, s2):
    p1, p2 = np.mean(s1), np.mean(s2)
    n1, n2 = s1.size, s2.size
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    if p == 1. or p == 0.:
        return 0., 0.99
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    p_val = (1 - norm.cdf(np.abs(z))) * 2
    return z, p_val

def write_results(test, state_sets, indices, consistencies=None):
    n = test.results[0].n_sessions
    trial_counter = 0
    state_dict = {state: {'sessions': [], 'trials': [], 'pmfs': []} for state in range(len(state_sets))}

    for state, trials in enumerate(state_sets):

        session_js, pmfs, pmf_weights, _ = state_pmfs(test, trials, indices)
        state_dict[state]['sessions'] = session_js
        state_dict[state]['pmfs'] = pmfs

    for seq_num in range(n):
        for state, trials in enumerate(state_sets):
            relevant_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(test.results[0].models[0].stateseqs[seq_num]))]
            active_trials = np.zeros(len(test.results[0].models[0].stateseqs[seq_num]), dtype=int)

            if consistencies is None:
                active_trials[relevant_trials - trial_counter] = 1
            else:
                active_trials[relevant_trials - trial_counter] = np.sum(consistencies[tuple(np.meshgrid(relevant_trials, trials))], axis=0)
                active_trials[relevant_trials - trial_counter] -= 1
                active_trials[relevant_trials - trial_counter] = active_trials[relevant_trials - trial_counter] / (trials.shape[0] - 1)

            state_dict[state]['trials'].append(active_trials)

        trial_counter += len(test.results[0].models[0].stateseqs[seq_num])

    session_dict = {seq_num: {} for seq_num in range(n)}
    for seq_num in range(n):
        n_trials = len(test.results[0].models[0].stateseqs[seq_num])
        session_dict[seq_num]['states'] = np.zeros((len(state_sets), n_trials))
        c_n_a = test.results[0].data[seq_num]
        session_dict[seq_num]['contrasts'] = all_conts[cont_mapping(- c_n_a[:, 0] + c_n_a[:, 1])]
        for state in range(len(state_sets)):
            session_dict[seq_num]['states'][state] = state_dict[state]['trials'][seq_num]

    return state_dict, session_dict

if __name__ == "__main__":

    fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][0]
    # if fit_type == 'bias':
    #     loading_info = json.load(open("canonical_infos_bias.json", 'r'))
    # elif fit_type == 'prebias':
    #     loading_info = json.load(open("canonical_infos.json", 'r'))
    # subjects = []
    # regexp = re.compile(r'canonical_result_((\w|-)+)_prebias((_var_0.03)*).p')
    # for filename in os.listdir("./multi_chain_saves/"):
    #     if not (filename.startswith('canonical_result_') and filename.endswith('.p')):
    #         continue
    #     result = regexp.search(filename)
    #     if result is None:
    #         continue
    #     subject = result.group(1)
    #     subjects.append(subject)

    subjects = ['MFD_09', 'PL037', 'MFD_08', 'UCLA035', 'CSH_ZAD_025', 'DY_011', 'KS052', 'SWC_052', 'UCLA049', 'NYU-37', 'ibl_witten_25', 'UCLA048', 'DY_010', 'CSHL054',
                'CSH_ZAD_024', 'UCLA034', 'KS046', 'KS094', 'KS044', 'KS096', 'PL034', 'NR_0028', 'CSH_ZAD_026', 'NYU-48', 'UCLA036', 'SWC_038', 'KS051',
                'ibl_witten_18', 'ibl_witten_27', 'ibl_witten_19', 'SWC_039', 'DY_013', 'CSHL_020', 'UCLA037', 'NR_0029', 'PL035', 'UCLA033', 'ZFM-01937', 'KS016',
                'CSHL053', 'KS086', 'PL024', 'ZFM-01576', 'ZFM-02370', 'SWC_054', 'NYU-30', 'CSHL047', 'ZFM-01577', 'KS055', 'CSHL052', 'KS017', 'DY_016', 'PL030',
                'ZFM-01936', 'CSH_ZAD_022', 'UCLA030', 'KS042', 'NYU-27', 'KS015', 'DY_014', 'NYU-65', 'ibl_witten_20', 'CSHL045', 'ZFM-02373', 'ZFM-02372', 'KS084',
                'CSHL051', 'KS014', 'SWC_043', 'ZFM-01935', 'KS043', 'KS091', 'PL033', 'KS022', 'CSHL059', 'NR_0019', 'SWC_022', 'NYU-47', 'NR_0027', 'CSH_ZAD_029',
                'ibl_witten_17', 'DY_008', 'SWC_060', 'ibl_witten_29', 'UCLA012', 'SWC_061', 'ibl_witten_16', 'DY_009', 'UCLA044', 'SWC_023', 'NYU-46',  'MFD_05',
                'CSHL058', 'NYU-11', 'KS023', 'KS021', 'DY_020', 'ZFM-05236', 'SWC_021', 'MFD_07', 'ibl_witten_14', 'NYU-06', 'NR_0031',
                'UCLA011', 'CSH_ZAD_001', 'ZFM-01592', 'CSHL_007', 'NYU-39', 'MFD_06', 'NYU-45', 'ZM_2245', 'UCLA005', 'UCLA052', 'PL050', 'NYU-12', 'ZFM-02369',
                'NR_0021', 'ZM_2241', 'CSH_ZAD_011', 'SWC_066', 'UCLA014', 'PL016', 'PL017', 'ZM_1897', 'NR_0020', 'ZM_2240', 'NYU-40', 'ZFM-02368',
                'CSHL060', 'KS019', 'DY_018', 'CSHL_015', 'CSHL049', 'UCLA017', 'PL015', 'ibl_witten_13', 'ZM_3003', 'CSHL_014', 'SWC_065']
    # worked: 
    
    # no data attribute: 'MFD_09', 
    # too big: 'UCLA006', 'NR_0024' 'UCLA015'
    print(len(subjects))
    fit_variance = 0.04
    dur = 'yes'

    in_sess_appear_n_bins = 20
    in_sess_appear = np.zeros((3, in_sess_appear_n_bins))
    in_train_appear_n_bins = 10
    in_train_appear = np.zeros((3, in_train_appear_n_bins))

    contrast_intro_types = []  # list to agglomorate in which state type which contrast is introduced
    intros_by_type_sum = np.zeros(3)  # array to agglomorate how many states where introduced during which type, normalised by length of phase

    not_yet = True

    abs_state_durs = []
    all_first_pmfs = {}
    all_first_pmfs_typeless = {}
    all_trial_ns = {}
    all_pmf_diffs = []
    all_pmf_asymms = []
    all_pmfs = []
    all_changing_pmfs = []
    all_changing_pmf_names = []
    all_intros = []
    all_intros_div = []
    all_states_per_type = []
    regressions = []
    regression_diffs = []
    all_pmf_weights = []
    all_weight_trajectories = []
    bias_sessions = []
    first_and_last_pmf = []
    all_pmfs_named = {}
    all_sudden_changes = [[], [], []]
    all_sudden_transition_changes = [[], []]
    aug_all_sudden_changes = [[], [], []]
    aug_all_sudden_transition_changes = [[], []]
    captured_states = []
    type_1_to_2_save = []
    session_time_at_sudden_changes = [[], []]
    all_state_percentages = []
    regressed_or_not_list = []
    regression_magnitude_list = []
    dates_list = []

    all_accs, all_pers = [], []

    new_counter, transform_counter = 0, 0
    state_types_interpolation = np.zeros((3, 150))
    all_state_types = []

    ultimate_counter = 0
    trial_counter = 0
    session_counter = 0

    just_checking = False
    checking_counter = 0

    works_counter = 0
    fail_counter = 0
    check_yes = []
    all_summed_beliefs = []

    print("WARNING, not saving")

    for subject in subjects[::2]:
        if subject.startswith('GLM_Sim_') or subject.startswith('fip_') or subject in ['PL037', 'PL034', 'PL035', 'DY_010', 'SWC_065']:
        #     # ibl_witten_18 is a weird one, super good session in the middle, ending phase 1, never to re-appear, bad at the end
        #     # ZFM-05245 is neuromodulator mouse, never reaches ephys it seems... same for ZFM-04019
        #     # SWC_065 never reaches type 3
        #     # UCLA006 is too large to load both the canonical result and its consistencies
        #     # 'NYU-12' has no data for some reason
        #     # UCLA015 has a problem
        # old skips: SWC_065', 'ZFM-05245', 'ZFM-04019', 'ibl_witten_18', 'NYU-12', 'UCLA015', 'CSHL062', 'CSHL_018', 'CSHL061'
        # new comments:
        # PL037 starts out perfect and has no contrast introduction, missing data
        # PL034 also perfect from start, missing data
        # PL035 same, missing data
        # DY_010 is weird, investigate: missing data
        # SWC_065 never reaches type 3
            continue

        print(subject)

        if just_checking:
            print(checking_counter)
        else:
            print(ultimate_counter)

        try:
            info_dict = pickle.load(open("./{}/{}_info_dict.p".format('session_data', subject), "rb"))
            if 'ephys_start' in info_dict:
                bias_sessions.append(info_dict['ephys_start'] - info_dict['bias_start'])
            else:
                bias_sessions.append(info_dict['n_sessions'] - info_dict['bias_start'])
                print(subject, info_dict['n_sessions'], info_dict['bias_start'])
            works_counter += 1
        except:
            info_dict = json.load(open("./{}/{}_info_dict.json".format('session_data', subject), 'r'))
            info_dict['dates'] = [datetime.datetime.fromisoformat(d) for d in info_dict['dates']]
            if 'ephys_start' in info_dict:
                bias_sessions.append(info_dict['ephys_start'] - info_dict['bias_start'])
            else:
                bias_sessions.append(info_dict['n_sessions'] - info_dict['bias_start'])
                print(subject, info_dict['n_sessions'], info_dict['bias_start'])
            

        print(subject)
        dates_list.append(info_dict['dates'][3:info_dict['bias_start'] + 3])
        # pickle.dump(dates_list, open("multi_chain_saves/dates_list.p", 'wb'))
        # pickle.dump(bias_sessions, open("multi_chain_saves/bias_sessions.p", 'wb'))

        mode_specifier = 'first'
        if os.path.exists("multi_chain_saves/canonical_result_{}_{}.p".format(subject, fit_type)):
            fit_var = False
        else:
            fit_var = True

        if just_checking:
            assert os.path.exists("multi_chain_saves/canonical_result_{}_{}.p".format(subject, fit_type)) or \
                   os.path.exists("multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance))
            assert os.path.exists("multi_chain_saves/{}_mode_indices_{}_{}".format(mode_specifier, subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p') or \
                   os.path.exists("multi_chain_saves/mode_indices_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p')
            assert os.path.exists("multi_chain_saves/{}_state_sets_{}_{}".format(mode_specifier, subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p') or \
                   os.path.exists("multi_chain_saves/state_sets_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p')
            assert os.path.exists("multi_chain_saves/first_mode_consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p') or \
                   os.path.exists("multi_chain_saves/consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p')
            checking_counter += 1
            check_yes.append(subject)
            continue

        test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        if len(test.results[0].infos) < len(info_dict):
            test.results[0].infos = info_dict
            test.results[0].n_contrasts = 11
            print(f'overwriting info dict of {subject}')
            pickle.dump(test, open("multi_chain_saves/canonical_result_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'wb'))
        try:
            mode_indices = pickle.load(open("multi_chain_saves/{}_mode_indices_{}_{}".format(mode_specifier, subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
            state_sets = pickle.load(open("multi_chain_saves/{}_state_sets_{}_{}".format(mode_specifier, subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        except Exception as E:
            print(E)
            try:
                mode_indices = pickle.load(open("multi_chain_saves/mode_indices_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
                state_sets = pickle.load(open("multi_chain_saves/state_sets_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
            except Exception as E:
                print(E)
                print("____________________________________")
                print("Something quite wrong with {}".format(subject))
                print("____________________________________")
                continue
        ultimate_counter += 1

        print(works_counter, fail_counter)

        # try:
        #     consistencies = pickle.load(open("multi_chain_saves/first_mode_consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        # except FileNotFoundError:
        #     consistencies = pickle.load(open("multi_chain_saves/consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        # consistencies /= consistencies[0, 0]

        # all_summed_beliefs += extrapolate_belief_strength(test, consistencies)

        # continue

        try:

            # n = test.results[0].n_sessions
            # trial_counter = 0
            # weight_collection = []

            # for seq_num in range(n):

            #     weight_collection.append(np.zeros(len(test.results[0].models[0].stateseqs[seq_num])))

            #     state_consistencies = {}  # save when which state is how certain, to later compare their responses
            #     for state, trials in enumerate(state_sets):

            #         relevant_trials = trials[np.logical_and(trial_counter <= trials, trials < trial_counter + len(test.results[0].models[0].stateseqs[seq_num]))]
            #         active_trials = np.zeros(len(test.results[0].models[0].stateseqs[seq_num]))

            #         active_trials[relevant_trials - trial_counter] = np.sum(consistencies[tuple(np.meshgrid(relevant_trials, trials))], axis=0)
            #         active_trials[relevant_trials - trial_counter] -= 1
            #         active_trials[relevant_trials - trial_counter] = active_trials[relevant_trials - trial_counter] / (trials.shape[0] - 1)

            #         if active_trials.sum() > 0:
            #             weight_collection[-1] += active_trials

            #     trial_counter += len(test.results[0].models[0].stateseqs[seq_num])

            state_assistance = False
            all_infos = predictive_check(test, mode_indices)
            pickle.dump(all_infos, open("./pred_checks_8/all_infos_{}_weighted".format(subject, state_assistance), 'wb'))
        except Exception as E:
            print(E)
            quit()
            continue
        continue

        states, pmfs, pmf_weights, durs, state_types, contrast_intro_type, intros_by_type, undiv_intros, states_per_type, trial_ns = state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save=True, show=False, separate_pmf=1, type_coloring=True, dpi=300, save_append=str(fit_variance).replace('.', '_'))
        # compare_pmfs(test, [0, 1], states, pmfs)
        
        try:
            consistencies = pickle.load(open("multi_chain_saves/first_mode_consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        except FileNotFoundError:
            consistencies = pickle.load(open("multi_chain_saves/consistencies_{}_{}".format(subject, fit_type) + fit_var * "_var_{}".format(fit_variance) + '.p', 'rb'))
        consistencies /= consistencies[0, 0]
        # contrasts_plot(test, [s for s in state_sets if len(s) > 40], dpi=300, subject=subject, save=True, show=False, consistencies=consistencies, CMF=False)
        # quit()

        basic_info, diffs, regressed_or_not, regression_magnitude = pmf_regressions(states, pmfs, durs)
        regressions.append(basic_info)
        regression_diffs.append(diffs)
        regressed_or_not_list.append(regressed_or_not)
        regression_magnitude_list.append(regression_magnitude)
        dates_list.append(info_dict['dates'][3:test.results[0].n_sessions + 3])

        trial_counter += test.results[0].n_datapoints
        session_counter += test.results[0].n_sessions

        print(trial_counter / ultimate_counter)
        print("trial and session counter", trial_counter, session_counter)
        captured_states.append((len([item for sublist in state_sets for item in sublist if len(sublist) > 40]), test.results[0].n_datapoints, len([s for s in state_sets if len(s) > 40]), test.results[0].n_sessions))

        # training overview
        # _ = state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save_append='step 0', show=1, separate_pmf=1, type_coloring=False, dont_plot=list(range(8)), plot_until=-1)
        # _ = state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save_append='step 1', show=1, separate_pmf=1, type_coloring=False, dont_plot=list(range(7)), plot_until=2)
        # _ = state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save_append='step 2', show=1, separate_pmf=1, type_coloring=False, dont_plot=list(range(6)), plot_until=7)
        # _ = state_development(test, [s for s in state_sets if len(s) > 40], mode_indices, save_append='step 3', show=1, separate_pmf=1, type_coloring=False, dont_plot=list(range(4)), plot_until=13)
        all_state_percentages.append(states)

        abs_state_durs.append(durs)

        # state_dict, session_dict = write_results(test, [s for s in state_sets if len(s) > 40], mode_indices)
        # pickle.dump(state_dict, open("sofiya_data/state_dict_{}".format(subject), 'wb'))
        # pickle.dump(session_dict, open("sofiya_data/session_dict_{}".format(subject), 'wb'))

        # Test KS014's session number 12 for difference between state 5 and 6
        if subject == 'KS014' and True:

            # dur dist plotting - maybe I shouldn't average over parameters?
            # from scipy.stats import nbinom
            # for state in range(15):
            #     a = dur_hists(test, [s for s in state_sets if len(s) > 40][state], mode_indices)
            #     mix = np.zeros(1000)
            #     for i in range(len(a)):
            #         mix += nbinom.pmf(np.arange(1000), a[i][0], 1 - a[i][1]) / len(a)
            #     plt.plot(mix)
            #     # plt.plot(nbinom.pmf(np.arange(1000), a.mean(0)[0], 1- a.mean(0)[1]))
            #     plt.title(len(state_sets) - test.state_mapping[state])
            #     plt.show()

            _, cnas, state_consistencies = contrasts_plot(test, [s for s in state_sets if len(s) > 40], dpi=300, subject=subject, save=True, show=True, consistencies=consistencies, CMF=False)
            cnas = cnas[0] # works if we only return one cnas, from sess 12
            # a 0 in cnas[:, -1] is a rightwards answer
            # a 1 in cnas[:, 0] is a rightwards trial
            a1 = cnas[np.logical_and(cnas[:, 0] == 1, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 0] == 1, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))

            a1 = cnas[np.logical_and(cnas[:, 0] == 0.987, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 0] == 0.987, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))


            a1 = cnas[np.logical_and(cnas[:, 0] == 0.848, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 0] == 0.848, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))

            a1 = cnas[np.logical_and(cnas[:, 1] == 0.848, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 1] == 0.848, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))

            a1 = cnas[np.logical_and(cnas[:, 1] == 0.987, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 1] == 0.987, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))

            a1 = cnas[np.logical_and(cnas[:, 1] == 1, state_consistencies[4] > 0.5), -1]
            a2 = cnas[np.logical_and(cnas[:, 1] == 1, state_consistencies[6] > 0.5), -1]
            print(a1.sum(), a1.size, a2.sum(), a2.size)
            print(two_sample_binom_test(a1, a2))

            states = np.concatenate([np.repeat([4], np.sum(state_consistencies[4] > 0.5)), np.repeat([6], np.sum(state_consistencies[6] > 0.5))])
            state_5 = state_consistencies[4] > 0.5
            state_6 = state_consistencies[6] > 0.5
            contrasts = np.concatenate([cnas[state_5, 0] - cnas[state_5, 1], cnas[state_6, 0] - cnas[state_6, 1]])
            anwers = np.concatenate([cnas[state_5, -1], cnas[state_6, -1]])

            df = pd.DataFrame({'states': states,
                       'contrasts': contrasts,
                       'responses': anwers})

            from statsmodels.formula.api import ols
            import statsmodels.api as sm

            model = ols('responses ~ C(states) + C(contrasts)', data=df).fit()
            print(sm.stats.anova_lm(model, typ=2))
            quit()

        # lost session_contrasts somehow
        test.results[0].session_contrasts = [np.unique(cont_mapping(d[:, 0] - d[:, 1])) for d in test.results[0].data]
        sudden_changes, sudden_transition_changes, aug_sudden_changes, aug_sudden_transition_changes, in_sess_appear_dist, in_train_appear_dist, in_sess_appear_dist_old, in_train_appear_dist_old, type_1_to_2, session_time_at_sudden_change = sudden_state_changes(test, [s for s in state_sets if len(s) > 40], consistencies=consistencies, pmf_weights=pmf_weights, pmfs=pmfs)
        session_time_at_sudden_changes[0] += session_time_at_sudden_change[0]
        session_time_at_sudden_changes[1] += session_time_at_sudden_change[1]
        type_1_to_2_save.append(type_1_to_2)

        in_sess_appear += in_sess_appear_dist
        in_train_appear += in_train_appear_dist
        # print(test.n_sessions, train_appear_dist)

        all_first_pmfs_typeless[subject] = []
        for pmf in pmfs:
            all_first_pmfs_typeless[subject].append((pmf[0], pmf[1][0]))
            all_pmfs.append(pmf)
        
        first_pmfs, changing_pmfs = get_first_pmfs(states, pmfs)
        for pmf in changing_pmfs:
            if type(pmf[0]) == int:
                continue
            all_changing_pmf_names.append(subject)
            all_changing_pmfs.append(pmf)
        
        all_first_pmfs[subject] = first_pmfs

        all_trial_ns[subject] = trial_ns

        new = type_2_appearance(states, pmfs)

        if new == 2:
            print('____________________________')
            print(subject)
            print('____________________________')
        if new == 1:
            new_counter += 1
        if new == 0:
            transform_counter += 1
        print(new_counter, transform_counter)

        first_and_last_pmf.append((pmf_weights[np.argmax(states[:, 0])][0], pmf_weights[np.argmax(states[:, -1])][-1]))
        quit()
        all_sudden_changes[0] += sudden_changes[0]
        all_sudden_changes[1] += sudden_changes[1]
        all_sudden_changes[2] += sudden_changes[2]
        all_sudden_transition_changes[0] += sudden_transition_changes[0]
        all_sudden_transition_changes[1] += sudden_transition_changes[1]
        aug_all_sudden_changes[0] += aug_sudden_changes[0]
        aug_all_sudden_changes[1] += aug_sudden_changes[1]
        aug_all_sudden_changes[2] += aug_sudden_changes[2]
        aug_all_sudden_transition_changes[0] += aug_sudden_transition_changes[0]
        aug_all_sudden_transition_changes[1] += aug_sudden_transition_changes[1]

        if new != len(sudden_transition_changes[0]):
            print("look into{}".format(subject))

        all_weight_trajectories += pmf_weights

        all_pmfs_named[subject] = pmfs

        state_types_interpolation[0] += np.interp(np.linspace(1, state_types.shape[1], 150), np.arange(1, 1 + state_types.shape[1]), state_types[0])
        state_types_interpolation[1] += np.interp(np.linspace(1, state_types.shape[1], 150), np.arange(1, 1 + state_types.shape[1]), state_types[1])
        state_types_interpolation[2] += np.interp(np.linspace(1, state_types.shape[1], 150), np.arange(1, 1 + state_types.shape[1]), state_types[2])

        all_pmf_weights += [item for sublist in pmf_weights for item in sublist]
        all_state_types.append(state_types)

        all_intros.append(undiv_intros)
        all_intros_div.append(intros_by_type)
        if states_per_type != []:
            all_states_per_type.append(states_per_type)
        
        intros_by_type_sum += intros_by_type
        for pmf in pmfs:
            all_pmfs.append(pmf)
            for p in pmf[1]:
                all_pmf_diffs.append(p[-1] - p[0])
                all_pmf_asymms.append(np.abs(p[0] + p[-1] - 1))
        contrast_intro_types.append(contrast_intro_type)

    if ultimate_counter > 10 and False:
        pickle.dump(all_first_pmfs, open("all_first_pmfs.p", 'wb'))
        pickle.dump(all_changing_pmfs, open("changing_pmfs.p", 'wb'))
        pickle.dump(all_changing_pmf_names, open("changing_pmf_names.p", 'wb'))
        pickle.dump(all_first_pmfs_typeless, open("all_first_pmfs_typeless.p", 'wb'))
        pickle.dump(all_intros, open("all_intros.p", 'wb'))
        pickle.dump(all_intros_div, open("all_intros_div.p", 'wb'))
        pickle.dump(all_pmfs, open("all_pmfs.p", 'wb'))
        pickle.dump(all_states_per_type, open("all_states_per_type.p", 'wb'))
        pickle.dump(regressions, open("regressions.p", 'wb'))
        pickle.dump(regression_diffs, open("regression_diffs.p", 'wb'))
        # pickle.dump(all_state_types, open("all_state_types.p", 'wb'))
        pickle.dump(all_pmf_weights, open("all_pmf_weights.p", 'wb'))
        pickle.dump(state_types_interpolation, open("state_types_interpolation.p", 'wb'))
        # pickle.dump(state_types_interpolation, open("state_types_interpolation_4_states.p", 'wb'))  # special version, might not want to use
        abs_state_durs = np.array(abs_state_durs)
        pickle.dump(abs_state_durs, open("multi_chain_saves/abs_state_durs.p", 'wb'))
        pickle.dump(all_weight_trajectories, open("multi_chain_saves/all_weight_trajectories.p", 'wb'))
        pickle.dump(bias_sessions, open("multi_chain_saves/bias_sessions.p", 'wb'))
        pickle.dump(all_pmfs_named, open("multi_chain_saves/all_pmfs_named.p", 'wb'))
        pickle.dump(first_and_last_pmf, open("multi_chain_saves/first_and_last_pmf.p", 'wb'))
        pickle.dump(all_sudden_changes, open("multi_chain_saves/all_sudden_changes.p", 'wb'))
        pickle.dump(aug_all_sudden_changes, open("multi_chain_saves/aug_all_sudden_changes.p", 'wb'))
        pickle.dump(all_sudden_transition_changes, open("multi_chain_saves/all_sudden_transition_changes.p", 'wb'))
        pickle.dump(aug_all_sudden_transition_changes, open("multi_chain_saves/aug_all_sudden_transition_changes.p", 'wb'))
        pickle.dump(in_sess_appear, open("multi_chain_saves/in_sess_appear.p", 'wb'))
        pickle.dump(in_train_appear, open("multi_chain_saves/in_train_appear.p", 'wb'))
        pickle.dump(all_trial_ns, open("all_trial_ns.p", 'wb'))
        pickle.dump(captured_states, open("captured_states.p", 'wb'))
        pickle.dump(type_1_to_2_save, open("multi_chain_saves/type_1_to_2_save.p", 'wb'))
        pickle.dump(all_state_percentages, open("multi_chain_saves/all_state_percentages.p", 'wb'))
        pickle.dump(session_time_at_sudden_changes, open("multi_chain_saves/session_time_at_sudden_changes.p", 'wb'))
        pickle.dump(regressed_or_not_list, open("multi_chain_saves/regressed_or_not_list.p", 'wb'))
        pickle.dump(regression_magnitude_list, open("multi_chain_saves/regression_magnitude_list.p", 'wb'))
        pickle.dump(dates_list, open("multi_chain_saves/dates_list.p", 'wb'))
        pickle.dump((trial_counter, session_counter), open("multi_chain_saves/trial_and_session_counter.p", 'wb'))

    print("Ultimate count is {}".format(ultimate_counter))


    if True:
        abs_state_durs = pickle.load(open("multi_chain_saves/abs_state_durs.p", 'rb'))
        bias_sessions = pickle.load(open("multi_chain_saves/bias_sessions.p", 'rb'))

        print("Median split type fractions")
        print(abs_state_durs[abs_state_durs.sum(1) <= np.median(abs_state_durs.sum(1))].mean(0) / abs_state_durs[abs_state_durs.sum(1) <= np.median(abs_state_durs.sum(1))].mean(0).sum(0))
        # [0.23041475 0.16820276 0.60138249]
        print(abs_state_durs[abs_state_durs.sum(1) > np.median(abs_state_durs.sum(1))].mean(0) / abs_state_durs[abs_state_durs.sum(1) > np.median(abs_state_durs.sum(1))].mean(0).sum(0))
        # [0.21607055 0.14453699 0.63939245]
        n_quantiles = 10
        for i in range(n_quantiles):
            limit_low, limit_high = np.quantile(abs_state_durs.sum(1), [i / n_quantiles, (i + 1) / n_quantiles])
            mask = np.logical_and(limit_low <= abs_state_durs.sum(1), abs_state_durs.sum(1) < limit_high)
            print(i, limit_low, limit_high, mask.sum())
            print(abs_state_durs[mask].mean(0) / abs_state_durs[mask].mean(0).sum(0))
            print()
        
        # extra figure showing how histograms of how many sessions all the mice spend in the different stages
        f, axs = plt.subplots(3, 1, figsize=(16 * 0.75, 9 * 0.75), sharex=True, sharey=True)

        axs[0].hist(abs_state_durs[:, 0], bins=np.linspace(0, abs_state_durs.max(), abs_state_durs.max()), color='grey')
        axs[1].hist(abs_state_durs[:, 1], bins=np.linspace(0, abs_state_durs.max(), abs_state_durs.max()), color='grey')
        axs[2].hist(abs_state_durs[:, 2], bins=np.linspace(0, abs_state_durs.max(), abs_state_durs.max()), color='grey')

        axs[2].set_xlabel("# of sessions", size=24)
        axs[1].set_ylabel("# of mice", size=24)

        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        plt.xlim(0, abs_state_durs.max())
        plt.tight_layout()
        plt.savefig("./summary_figures/sessions_in_stages.png", dpi=300)
        plt.close()

        # compute all the needed correlations
        print("Correlations")
        from scipy.stats import pearsonr
        print(pearsonr(abs_state_durs[:, 0], abs_state_durs[:, 1]))
        # (0.30674757757240906, 0.0006911035612003034)
        print(pearsonr(abs_state_durs[:, 0], abs_state_durs[:, 2]))
        # (-0.004498545297409465, 0.9612734601349902)
        print(pearsonr(abs_state_durs[:, 1], abs_state_durs[:, 2]))
        # (0.271132413685335, 0.0028584516088162267)

        print(pearsonr(abs_state_durs[:, 0], bias_sessions))
        # (-0.08461110126496008, 0.3602453999133687)
        print(pearsonr(abs_state_durs[:, 1], bias_sessions))
        # (-0.11310414793889537, 0.22067640082120177)
        print(pearsonr(abs_state_durs[:, 2], bias_sessions))
        # (-0.09816405019391666, 0.28818669926511503)
        print(pearsonr(abs_state_durs.sum(1), bias_sessions))
        # (-0.13408773681373676, 0.14598537551184568)

        # 31.12.24
        # print(pearsonr(abs_state_durs[:, 0], abs_state_durs[:, 1]))
        # (0.2096921108928549, 0.015027324108210532)
        # print(pearsonr(abs_state_durs[:, 0], abs_state_durs[:, 2]))
        # # (0.03537942172848045, 0.684862647688938)
        # print(pearsonr(abs_state_durs[:, 1], abs_state_durs[:, 2]))
        # # (0.1446744640125377, 0.09535054971185515)
        # print(pearsonr(abs_state_durs[:, 0], bias_sessions))
        # # (-0.06241808976087822, 0.47370029857492496)
        # print(pearsonr(abs_state_durs[:, 1], bias_sessions))
        # # (-0.004761837777311942, 0.9564522442975754)
        # print(pearsonr(abs_state_durs[:, 2], bias_sessions))
        # # (-0.19989942279657802, 0.020572005514517072)
        # print(pearsonr(abs_state_durs.sum(1), bias_sessions))
        # # (-0.19178437238090368, 0.02642455352734737)


        # simplex plotting, figure 5
        from simplex_plot import plotSimplex
        plotSimplex(np.array(abs_state_durs), facecolors='none', edgecolors='k', linewidths=1.5, show=True, vertexcolors=[type2color[i] for i in range(3)], vertexlabels=['Stage 1', 'Stage 2', 'Stage 3'])

        # simplex inset
        plt.hist(abs_state_durs.sum(1), color='grey', bins=12)
        sns.despine()
        plt.xticks(size=26)
        plt.yticks(size=26)
        plt.ylabel("# of mice", size=40)
        plt.xlabel('# of sessions', size=40)
        plt.tight_layout()
        plt.savefig("./summary_figures/session_num_hist.png", dpi=300, transparent=True)
        plt.show()

        # simplex inset 2?
        captured_states = pickle.load(open("captured_states.p", 'rb'))
        num_states = np.array([x for _, _, x, _ in captured_states])

        plt.hist(num_states, color='grey', bins=np.arange(1, 16), align='left')
        plt.xlim(0.5, 15.5)
        sns.despine()
        plt.xticks(size=26)
        plt.yticks(size=26)
        plt.ylabel("# of mice", size=40)
        plt.xlabel('# of states', size=40)
        plt.tight_layout()
        plt.savefig("./summary_figures/state_num_hist.png", dpi=300, transparent=True)
        plt.show()

    if True:  # code for figure 7
        in_sess_appear = pickle.load(open("multi_chain_saves/in_sess_appear.p", 'rb'))
        in_train_appear = pickle.load(open("multi_chain_saves/in_train_appear.p", 'rb'))
        f, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        gs = axs[0, 0].get_subplotspec().get_gridspec()

        # clear the left column for the subfigure:
        for a in axs[:, 0]:
            a.remove()

        ax, ax2 = axs[0, 1], axs[1, 1]
        # plot the same data on both axes
        for i in range(in_sess_appear.shape[0]):
            ax.bar(np.linspace(0, 1, in_sess_appear_n_bins + 1)[:-1], in_sess_appear[i], bottom=in_sess_appear[:i].sum(0), align='edge', width=1/in_sess_appear_n_bins, color=type2color[i])
            ax2.bar(np.linspace(0, 1, in_sess_appear_n_bins + 1)[:-1], in_sess_appear[i], bottom=in_sess_appear[:i].sum(0), align='edge', width=1/in_sess_appear_n_bins, color=type2color[i])

        # zoom-in / limit the view to different portions of the data
        ax.set_ylim(350, 440)
        ax2.set_ylim(0, 35)

        ax.set_yticks([350, 400])
        ax.set_yticklabels([350, 400])
        ax2.set_yticks([0, 10, 20, 30])
        ax2.set_xticks([0, .25, .5, .75, 1])
        ax2.set_yticklabels([0, 10, 20, 30])
        ax2.set_xticklabels([0, .25, .5, .75, 1])

        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax.tick_params(bottom=False)

        ax2.set_xlim(left=0, right=1)
        ax2.set_xlabel('Session time at appearance', fontsize=34)
        ax2.tick_params(axis='both', labelsize=20)
        ax.tick_params(axis='both', labelsize=20)

        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        subfig = f.add_subfigure(gs[:, 0])
        a1 = subfig.subplots(1, 1)

        for i in range(in_train_appear.shape[0]):
            a1.bar(np.linspace(0, 1, in_train_appear_n_bins + 1)[:-1], in_train_appear[i], bottom=in_train_appear[:i].sum(0), align='edge', width=1/in_train_appear_n_bins, color=type2color[i])
        a1.set_xlim(left=0, right=1)
        # plt.title('First appearence of ', fontsize=22)
        a1.set_ylabel('# of states', fontsize=34)
        a1.set_xlabel('Training time at appearance', fontsize=34)
        a1.tick_params(axis='both', labelsize=20)
        a1.set_xticks([0, .25, .5, .75, 1])
        a1.set_xticklabels([0, .25, .5, .75, 1])
        a1.spines['top'].set_visible(False)
        a1.spines['right'].set_visible(False)

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="5%", height="100%", loc=9)
        for i in range(in_sess_appear.shape[0]):
            axins.bar([0], in_sess_appear[i, 0], bottom=in_sess_appear[:i, 0].sum(0), align='edge', color=type2color[i])
        axins.spines['right'].set_visible(False)
        # axins.spines['left'].set_visible(False)
        axins.spines['top'].set_visible(False)
        # axins.spines['bottom'].set_visible(False)
        axins.set_yticks([0, 100, 200, 300, 400])
        axins.set_yticklabels([0, 100, 200, 300, 400])
        axins.set_xticks([])
        axins.tick_params(axis='y', labelsize=20)

        ax.annotate("", xytext=(0.09, 355), xy=(0.35, 374), xycoords='data', arrowprops=dict(facecolor='black', headwidth=10, headlength=10, width=2))#, arrowprops=dict(arrowstyle="-|>,head_width=50,head_length=50", color='k'))

        plt.tight_layout()
        plt.savefig('./summary_figures/states in sessions', dpi=300)
        plt.show()
