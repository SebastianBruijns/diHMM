"""
   Code for creating figure 6, 12, A16, and A17 (and many slighly variations on them)
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
from analysis_pmf import pmf_type, type2color
from mpl_toolkits import mplot3d
from matplotlib.patches import ConnectionPatch
from scipy.stats import sem

contrasts_L = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])
contrasts_R = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])[::-1]

def weights_to_pmf(weights, with_bias=1):
    psi = weights[0] * contrasts_R + weights[1] * contrasts_L + with_bias * weights[-1]
    return 1 / (1 + np.exp(psi))

# this will show an additional metric inferred from the weights, the range of the PMF (distance between min and max of PMF)
show_weight_augmentations = False

all_weight_trajectories = pickle.load(open("multi_chain_saves/all_weight_trajectories.p", 'rb'))
first_and_last_pmf = np.array(pickle.load(open("multi_chain_saves/first_and_last_pmf.p", 'rb')))

all_sudden_changes = pickle.load(open("multi_chain_saves/all_sudden_changes.p", 'rb'))
all_sudden_transition_changes = pickle.load(open("multi_chain_saves/all_sudden_transition_changes.p", 'rb'))

aug_all_sudden_changes = pickle.load(open("multi_chain_saves/aug_all_sudden_changes.p", 'rb'))
aug_all_sudden_transition_changes = pickle.load(open("multi_chain_saves/aug_all_sudden_transition_changes.p", 'rb'))

performance_points = np.array([-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0])
reduced_points = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
weight_colours = ['blue', 'red', 'green', 'goldenrod', 'darkorange']
weight_colours_aug = ['blue', 'red', 'green', 'goldenrod', 'darkorange', 'purple']
ylabels = ["Cont left", "Cont right", "Persevere", "Bias left", "Bias right"]
ylabels_aug = ["Cont left", "Cont right", "Persevere", "Bias left", "Bias right", "PMF span"]

folder = "./reward_analysis/"

local_ylabels = ylabels_aug if show_weight_augmentations else ylabels
local_weight_colours = weight_colours_aug if show_weight_augmentations else weight_colours

n_weights = all_weight_trajectories[0][0].shape[0]
n_types = 3


def create_nested_list(list_of_ns):
    """
    Create a complex nested list, according to the specifications of list_of_ns.
    E.g. [3, 2, 4] will return a list with 3 sublists, each containing 2 sublists again, each containing 4 sublists yet again.
    """
    ret = []
    if len(list_of_ns) == 0:
        return ret

    for _ in range(list_of_ns[0]):
        ret.append(create_nested_list(list_of_ns=list_of_ns[1:]))

    return ret


def pmf_to_perf(pmf):
    # determine performance of a pmf, but only on the omnipresent strongest contrasts
    return np.mean(np.abs(performance_points[reduced_points] + pmf[reduced_points]))


def pmf_type_rew(weights):
    rew = pmf_to_perf(weights_to_pmf(weights))
    if rew < 0.6:
        return 0
    elif rew < 0.7827:
        return 1
    else:
        return 2


def plot_traces_and_collate_data(data, augmented_data=None, title=""):
    """Plot all the individual weight change traces, ann collect the data we need for the other plots"""
    sudden_changes = len(data) == n_types - 1
    average = np.zeros((n_weights + 1 + show_weight_augmentations, n_types - sudden_changes, 2))
    counter = np.zeros((n_weights + 1 + show_weight_augmentations, n_types - sudden_changes))
    all_datapoints = create_nested_list([n_weights + 1 + show_weight_augmentations, n_types - sudden_changes, 2])

    f, axs = plt.subplots(n_weights + 1 + show_weight_augmentations, n_types - sudden_changes, figsize=(4 * (3 - sudden_changes), 9))  # We add another weight slot to split the bias
    for state_type, weight_gaps in enumerate(data):

        for gap in weight_gaps:
            for i in range(n_weights):
                if i == n_weights - 1:
                    axs[i + (gap[1][i] > 0), state_type].plot([0, 1], [gap[0][i], gap[-1][i]], marker="o")  # plot how the weight of the new state differs from the previous closest weight
                    average[i + (gap[1][i] > 0), state_type] += np.array([gap[0][i], gap[-1][i]])
                    counter[i + (gap[1][i] > 0), state_type] += 1
                    all_datapoints[i + (gap[1][i] > 0)][state_type][0].append(gap[0][i])
                    all_datapoints[i + (gap[1][i] > 0)][state_type][1].append(gap[-1][i])
                else:
                    axs[i, state_type].plot([0, 1], [gap[0][i], gap[-1][i]], marker="o")  # plot how the weight of the new state differs from the previous closest weight
                    average[i, state_type] += np.array([gap[0][i], gap[-1][i]])
                    counter[i, state_type] += 1
                    all_datapoints[i][state_type][0].append(gap[0][i])
                    all_datapoints[i][state_type][1].append(gap[-1][i])

        if show_weight_augmentations:
            for state_type, weight_gaps in enumerate(augmented_data):
                for gap in weight_gaps:
                    axs[n_weights + 1, state_type].plot([0, 1], [gap[0], gap[-1]], marker="o")  # plot how the weight of the new state differs from the previous closest weight
                    average[n_weights + 1, state_type] += np.array([gap[0], gap[-1]])
                    counter[n_weights + 1, state_type] += 1
                    all_datapoints[n_weights + 1][state_type][0].append(gap[0])
                    all_datapoints[n_weights + 1][state_type][1].append(gap[-1])

    plt.tight_layout()
    plt.savefig("./summary_figures/weight_changes/" + title + " augmented" * show_weight_augmentations)
    plt.close()

    return average, counter, all_datapoints


def plot_compact(average, counter, title, show_first_and_last=False, show_weight_augmentations=False, all_datapoints=[]):
    """Plot the means of the weight change traces, split by the different weight types, possibly with augmentations"""
    sudden_changes = average.shape[1] == n_types - 1
    f, axs = plt.subplots(1, n_types - sudden_changes, figsize=(4 * (3 - sudden_changes), 6))
    for i in range(n_types - sudden_changes):
        for j in range(n_weights + 1 + show_weight_augmentations):
            axs[i].plot([0, 1], average[j, i] / counter[j, i], marker="o", color=local_weight_colours[j], label=local_ylabels[j])
            axs[i].set_ylim(-3.5, 3.5)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([])
            if i == 0 and show_first_and_last:
                axs[i].set_ylabel("Weights", size=24)
                if j < n_weights - 1:
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[:, 0, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very first state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 0, -1] < 0
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 0, -1] > 0
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])
            else:
                axs[i].yaxis.set_ticklabels([])
            if i == 2 and show_first_and_last:
                if j < n_weights - 1:
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[:, 1, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very last state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 1, -1] < 0
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 1, -1] > 0
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])
            if j == 0:
                axs[i].set_title("Type {}".format(i + 1 + sudden_changes), size=26)
            if j == n_weights and i == 1:
                axs[i].set_xlabel("Lifetime weight change", size=24)
    axs[0].legend(frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig("./summary_figures/weight_changes/" + title + " augmented" * show_weight_augmentations)
    plt.close()


def plot_compact_all(average_slow, counter_slow, average_sudden, counter_sudden, title, all_data_sudden=[], all_data_slow=[]):
    """Plot a whole bunch of changes"""
    titles = ["Slow process\ntype 1", r'Type $1 \rightarrow 2$', "Slow process\ntype 2", r'Type $2 \rightarrow 3$', "Slow process\ntype 3"]
    f, axs = plt.subplots(1, 5, width_ratios=[1.1, 1, 1, 1, 1.1], figsize=(4 * 5, 8))
    average = np.zeros((average_slow.shape[0], average_slow.shape[1] + average_sudden.shape[1], 2))
    counter = np.zeros((counter_slow.shape[0], counter_slow.shape[1] + counter_sudden.shape[1]))
    all_data = create_nested_list([n_weights + 1, average_slow.shape[1] + average_sudden.shape[1], 2])
    average[:, [0, 2, 4]] = average_slow
    average[:, [1, 3]] = average_sudden
    counter[:, [0, 2, 4]] = counter_slow
    counter[:, [1, 3]] = counter_sudden
    # all_data[:, [0, 2, 4]] = counter_slow # this won't just work...
    # all_data[:, [1, 3]] = counter_sudden
    offset = 0.008
    for i in range(average.shape[1]):
        for j in range(n_weights + 1 + show_weight_augmentations):
            axs[i].plot([0 + (2- j) * offset, 1 + (2- j) * offset], average[j, i] / counter[j, i], marker="o", color=local_weight_colours[j], label=local_ylabels[j])
            if i == 0:
                local_data = all_data_slow[j][0]
            elif i == 2:
                local_data = all_data_slow[j][1]
            elif i == 4:
                local_data = all_data_slow[j][2]
            elif i == 1:
                local_data = all_data_sudden[j][0]
            elif i == 3:
                local_data = all_data_sudden[j][1]

            assert np.allclose(average[j, i] / counter[j, i], np.array([np.mean(local_data[0]), np.mean(local_data[1])]))
            axs[i].plot([0 + (2- j) * offset, 0 + (2- j) * offset], [average[j, i][0] / counter[j, i] - sem(local_data[0]), average[j, i][0] / counter[j, i] + sem(local_data[0])], color=local_weight_colours[j])
            axs[i].plot([1 + (2- j) * offset, 1 + (2- j) * offset], [average[j, i][1] / counter[j, i] - sem(local_data[1]), average[j, i][1] / counter[j, i] + sem(local_data[1])], color=local_weight_colours[j])
            
            axs[i].set_ylim(-3.5, 3.5)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([])
            if i == 0:
                axs[i].set_ylabel("Weights", size=38)
                if j < n_weights - 1:
                    axs[i].plot([-0.1], [np.mean(first_and_last_pmf[:, 0, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very first state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 0, -1] < 0
                    axs[i].plot([-0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 0, -1] > 0
                    axs[i].plot([-0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])
            else:
                axs[i].yaxis.set_ticklabels([])
            if i == 4:
                if j < n_weights - 1:
                    axs[i].plot([1.1], [np.mean(first_and_last_pmf[:, 1, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very last state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 1, -1] < 0
                    axs[i].plot([1.1], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 1, -1] > 0
                    axs[i].plot([1.1], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])
            axs[i].annotate("n={}".format(int(counter[0, i])), (0.06, 0.025), xycoords='axes fraction', size=26)
            if j == 0:
                axs[i].set_title(titles[i], size=34)
            if j == n_weights and i == 2:
                axs[i].set_xlabel("Weight change", size=38)
    axs[0].legend(frameon=False, fontsize=17)
    plt.tight_layout()
    plt.savefig("./summary_figures/weight_changes/" + title + " augmented" * show_weight_augmentations, dpi=300)
    plt.close()


def plot_compact_split(all_datapoints, title, show_first_and_last=False, show_weight_augmentations=False, width_divisor=20):
    """Plot the means of the weight change traces, split by the different weight types, possibly with augmentations"""
    sudden_changes = len(all_datapoints[0]) == 2
    f, axs = plt.subplots(1, n_types - sudden_changes, figsize=(4 * (3 - sudden_changes), 6))
    for i in range(n_types - sudden_changes):
        for j in range(n_weights + 1 + show_weight_augmentations):
            after, before = np.array(all_datapoints[j][i][1]), np.array(all_datapoints[j][i][0])
            deltas = after - before
            mask = deltas >= 0
            axs[i].plot([0, 1], [np.mean(before[mask]), np.mean(after[mask])], marker="o", color=local_weight_colours[j], label=local_ylabels[j], lw=mask.sum() / width_divisor)
            axs[i].plot([0, 1], [np.mean(before[~mask]), np.mean(after[~mask])], marker="o", color=local_weight_colours[j], ls='--', lw=(~mask).sum() / width_divisor)
            axs[i].set_ylim(-3.5, 3.5)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([])
            if i == 0 and show_first_and_last:
                axs[i].set_ylabel("Weights", size=24)
                if j < n_weights - 1:
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[:, 0, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very first state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 0, -1] < 0
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 0, -1] > 0
                    axs[i].plot([0.1], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', color=local_weight_colours[j])
            else:
                axs[i].yaxis.set_ticklabels([])
            if i == 2 and show_first_and_last:
                if j < n_weights - 1:
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[:, 1, j])], marker='*', color=local_weight_colours[j])  # also plot weights of very last state average
                if j == n_weights - 1:
                    mask = first_and_last_pmf[:, 1, -1] < 0
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])  # separete biases again
                if j == n_weights:
                    mask = first_and_last_pmf[:, 1, -1] > 0
                    axs[i].plot([0.9], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', color=local_weight_colours[j])
            if j == 0:
                axs[i].set_title("Type {}".format(i + 1 + sudden_changes), size=26)
            if j == n_weights and i == 1:
                axs[i].set_xlabel("Lifetime weight change", size=24)
    axs[0].legend(frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig("./summary_figures/weight_changes/split " + title + " augmented" * show_weight_augmentations)
    plt.close()


def plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal, x_lim_used_bias, bin_sets, title, x_lim_used_augment=0, show_deltas=True, show_first_and_last=False, show_weight_augmentations=False):
    """Plot histograms over the weights, and the mean changes connecting them.
    Might have to mess quite a bit with the y-axis"""
    sudden_changes = len(all_datapoints[0]) == 2
    x_steps = x_lim_used_bias - x_lim_used_bias % 5
    delta_dists = {}  # save the deltas, to see whether bias moves more in sudden changes

    f, axs = plt.subplots(n_weights + 1 + show_weight_augmentations, (n_types - sudden_changes) * 2, figsize=(4 * (3 - sudden_changes), 9))
    for i in range(n_types - sudden_changes):
        for j in range(n_weights + 1 + show_weight_augmentations):
            if j < 2:
                bins = bin_sets[0]
            elif j == 2:
                bins = bin_sets[1]
            elif j in [3, 4]:
                bins = bin_sets[2]
            else:
                bins = bin_sets[3]

            means = average[j, i] / counter[j, i]
            axs[j, i * 2].hist(all_datapoints[j][i][0], orientation='horizontal', bins=bins, color='grey', alpha=0.5)
            if show_deltas:
                delta_dists[(i, j)] = np.array(all_datapoints[j][i][1]) - np.array(all_datapoints[j][i][0])
                axs[j, i * 2 + 1].hist(np.array(all_datapoints[j][i][1]) - np.array(all_datapoints[j][i][0]), orientation='horizontal', bins=bins, color='red', alpha=0.5)
            else:
                axs[j, i * 2 + 1].hist(all_datapoints[j][i][1], orientation='horizontal', bins=bins, color='grey', alpha=0.5)

            axs[j, i * 2].set_ylim(bins[0], bins[-1])
            axs[j, i * 2 + 1].set_ylim(bins[0], bins[-1])

            axs[j, i * 2].spines['top'].set_visible(False)
            axs[j, i * 2].spines['right'].set_visible(False)
            # axs[j, i * 2].set_xticks([])
            axs[j, i * 2 + 1].spines['top'].set_visible(False)
            axs[j, i * 2 + 1].spines['right'].set_visible(False)
            # axs[j, i * 2 + 1].set_xticks([])

            # axs[j, i * 2].annotate("Var {:.2f}".format(np.var(all_datapoints[j][i][0])), xy=(0.65, 0.8), xycoords='axes fraction')
            # if show_deltas:
            #     axs[j, i * 2 + 1].annotate("Var {:.2f}".format(np.var(np.array(all_datapoints[j][i][1]) - np.array(all_datapoints[j][i][0]))), xy=(0.65, 0.8), xycoords='axes fraction')
            # else:
            #     axs[j, i * 2 + 1].annotate("Var {:.2f}".format(np.var(all_datapoints[j][i][1])), xy=(0.65, 0.8), xycoords='axes fraction')

            if j < n_weights - 1:
                assert x_lim_used_normal > max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]), "Hists are cut off ({} vs {})".format(x_lim_used_normal, max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]))
                axs[j, i * 2].set_xlim(0, x_lim_used_normal)
                axs[j, i * 2 + 1].set_xlim(0, x_lim_used_normal)
            elif j < n_weights + 1:
                assert x_lim_used_bias > max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]), "Hists are cut off ({} vs {})".format(x_lim_used_bias, max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]))
                axs[j, i * 2].set_xlim(0, x_lim_used_bias)
                axs[j, i * 2 + 1].set_xlim(0, x_lim_used_bias)
            else:
                assert x_lim_used_augment > max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]), "Hists are cut off ({} vs {})".format(x_lim_used_augment, max(axs[j, i * 2].set_xlim()[0], axs[j, i * 2 + 1].set_xlim()[1]))
                axs[j, i * 2].set_xlim(0, x_lim_used_augment)
                axs[j, i * 2 + 1].set_xlim(0, x_lim_used_augment)
            con = ConnectionPatch(xyA=(0, means[0]), xyB=(0, means[1] - show_deltas * means[0]), coordsA="data", coordsB="data",
                                    axesA=axs[j, i * 2], axesB=axs[j, i * 2 + 1], color="blue")
            axs[j, i * 2 + 1].add_artist(con)

            if i == 0:
                axs[j, i].set_ylabel(local_ylabels[j], size=15)
                # axs[j, i * 2 + 1].yaxis.set_ticklabels([])
                if show_first_and_last:
                    if j < n_weights - 1:
                        axs[j, i * 2].plot([x_lim_used_normal / 8], [np.mean(first_and_last_pmf[:, 0, j])], marker='*', c=local_weight_colours[j])  # also plot weights of very first state average
                    elif j == n_weights - 1:
                        mask = first_and_last_pmf[:, 0, -1] < 0
                        axs[j, i * 2].plot([x_lim_used_bias / 8], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', c=local_weight_colours[j])  # separete biases again
                    elif j == n_weights:
                        mask = first_and_last_pmf[:, 0, -1] > 0
                        axs[j, i * 2].plot([x_lim_used_bias / 8], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', c=local_weight_colours[j])
            # else:
            #     axs[j, i * 2].yaxis.set_ticklabels([])
            #     axs[j, i * 2 + 1].yaxis.set_ticklabels([])
            if i == n_types - 1 and show_first_and_last:
                if j < n_weights - 1:
                    axs[j, i * 2 + 1].plot([x_lim_used_normal / 8 * 7], [np.mean(first_and_last_pmf[:, 1, j]) - show_deltas * means[0]], marker='*', c=local_weight_colours[j])  # also plot weights of very last state average
                elif j == n_weights - 1:
                    mask = first_and_last_pmf[:, 1, -1] < 0
                    axs[j, i * 2 + 1].plot([x_lim_used_bias / 8 * 7], [np.mean(first_and_last_pmf[mask, 1, -1]) - show_deltas * means[0]], marker='*', c=local_weight_colours[j])  # separete biases again
                elif j == n_weights:
                    mask = first_and_last_pmf[:, 1, -1] > 0
                    axs[j, i * 2 + 1].plot([x_lim_used_bias / 8 * 7], [np.mean(first_and_last_pmf[mask, 1, -1]) - show_deltas * means[0]], marker='*', c=local_weight_colours[j])
            if j == 0:
                if 'sudden' in title:
                    axs[j, i * 2].set_title(r'Type ${} \rightarrow {}$'.format(i + 1, i + 2), loc='right', size=16, position=(1.45, 1))
                else:
                    axs[j, i * 2].set_title(r'Type ${}$'.format(i + 1), loc='right', size=16, position=(1.45, 1))
            if j == n_weights + show_weight_augmentations and i == 0:
                axs[j, 0].set_xlabel("Pre", size=15)
                if show_deltas:
                    axs[j, 1].set_xlabel("Deltas", size=15)
                else:
                    axs[j, 1].set_xlabel("Post", size=15)

            if j < n_weights - 1:
                axs[j, i * 2].set_xticks(list(range(x_steps, x_lim_used_normal, x_steps)))
                axs[j, i * 2].set_xticklabels([])
                axs[j, i * 2 + 1].set_xticks(list(range(x_steps, x_lim_used_normal, x_steps)))
                axs[j, i * 2 + 1].set_xticklabels([])
            elif j < n_weights + 1:
                axs[j, i * 2].set_xticks(list(range(x_steps, x_lim_used_bias, x_steps)) if len(list(range(x_steps, x_lim_used_bias, x_steps))) >= 1 else [x_steps])
                axs[j, i * 2 + 1].set_xticks(list(range(x_steps, x_lim_used_bias, x_steps)) if len(list(range(x_steps, x_lim_used_bias, x_steps))) >= 1 else [x_steps])
                if j == n_weights + show_weight_augmentations:
                    axs[j, i * 2].set_xticklabels(list(range(x_steps, x_lim_used_bias, x_steps)) if len(list(range(x_steps, x_lim_used_bias, x_steps))) >= 1 else [x_steps])
                    axs[j, i * 2 + 1].set_xticklabels(list(range(x_steps, x_lim_used_bias, x_steps)) if len(list(range(x_steps, x_lim_used_bias, x_steps))) >= 1 else [x_steps])
                else:
                    axs[j, i * 2].set_xticklabels([])
                    axs[j, i * 2 + 1].set_xticklabels([])
            else:
                axs[j, i * 2].set_xticks(list(range(x_steps, x_lim_used_augment, x_steps)))
                axs[j, i * 2 + 1].set_xticks(list(range(x_steps, x_lim_used_augment, x_steps)))
                if j == n_weights + show_weight_augmentations:
                    axs[j, i * 2].set_xticklabels(list(range(x_steps, x_lim_used_augment, x_steps)))
                    axs[j, i * 2 + 1].set_xticklabels(list(range(x_steps, x_lim_used_augment, x_steps)))
                else:
                    axs[j, i * 2].set_xticklabels([])
                    axs[j, i * 2 + 1].set_xticklabels([])
            if show_deltas:
                axs[j, i * 2 + 1].set_ylim(bins[0] - means[0], bins[-1] - means[0])

    # plt.tight_layout()
    plt.savefig("./summary_figures/weight_changes/" + title, dpi=300)
    plt.close()

    return delta_dists


if True:

    bin_sets = [np.linspace(-6.5, 6.5, 30), np.linspace(-1, 2, 30), np.linspace(-3.5, 3.5, 30), np.linspace(-0.2, 1, 30)]  # TODO: make show_augmentation robuse

    average_sudden, counter_sudden, all_data_sudden = plot_traces_and_collate_data(data=all_sudden_transition_changes, augmented_data=aug_all_sudden_transition_changes, title="sudden_weight_change at transitions")

    average_slow = np.zeros((n_weights + 1, n_types, 2))
    counter_slow = np.zeros((n_weights + 1, n_types))
    all_data_slow = create_nested_list([n_weights + 1, n_types, 2])

    for weight_traj in all_weight_trajectories:

        if len(weight_traj) < 5 or len(weight_traj) > 15: # take out too short trajectories, and too long ones
            continue

        state_type = pmf_type(weights_to_pmf(weight_traj[0]))

        for i in range(n_weights):
            if i == n_weights - 1:
                average_slow[i + (weight_traj[0][i] > 0), state_type] += np.array([weight_traj[0][i], weight_traj[-1][i]])
                counter_slow[i + (weight_traj[0][i] > 0), state_type] += 1
                all_data_slow[i + (weight_traj[0][i] > 0)][state_type][0].append(weight_traj[0][i])
                all_data_slow[i + (weight_traj[0][i] > 0)][state_type][1].append(weight_traj[-1][i])
            else:
                average_slow[i, state_type] += np.array([weight_traj[0][i], weight_traj[-1][i]])
                counter_slow[i, state_type] += 1
                all_data_slow[i][state_type][0].append(weight_traj[0][i])
                all_data_slow[i][state_type][1].append(weight_traj[-1][i])

    temp_sudden_counter, switch, nonswitch = 0, 0, 0
    for temp_sudden_change in all_sudden_transition_changes[0]:
        prev_relevant_points, post_relevant_points = weights_to_pmf(temp_sudden_change[0])[[0, 1, -2, -1]], weights_to_pmf(temp_sudden_change[1])[[0, 1, -2, -1]]
        # print((1 - post_relevant_points[0] + 1 - post_relevant_points[1] + post_relevant_points[-2] + post_relevant_points[-1]) / 4)
        if np.abs(np.mean(post_relevant_points) - 0.5) < 0.05 or np.abs(np.mean(prev_relevant_points) - 0.5) < 0.05:
            # print("skipped " + str(prev_relevant_points) + " " + str(post_relevant_points))
            continue
        # print("considered " + str(prev_relevant_points) + " " + str(post_relevant_points))
        temp_sudden_counter += 1
        if (np.mean(prev_relevant_points) > 0.5 and np.mean(post_relevant_points) < 0.5) or (np.mean(prev_relevant_points) < 0.5 and np.mean(post_relevant_points) > 0.5):
            switch += 1
        else:
            nonswitch += 1
    print(switch, nonswitch)

    plot_compact_all(average_slow, counter_slow, average_sudden, counter_sudden, title="all weight changes combined", all_data_sudden=all_data_sudden, all_data_slow=all_data_slow)

    average, counter, all_datapoints = plot_traces_and_collate_data(data=all_sudden_transition_changes, augmented_data=aug_all_sudden_transition_changes, title="sudden_weight_change at transitions")

    plot_compact(average, counter, title="compact sudden_weight_change at transitions", show_weight_augmentations=show_weight_augmentations, all_datapoints=all_datapoints)

    plot_compact_split(all_datapoints, title="compact sudden_weight_change at transitions", show_weight_augmentations=show_weight_augmentations, width_divisor=18)

    plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=23, x_lim_used_bias=11, x_lim_used_augment=40, bin_sets=bin_sets, title="weight changes sudden at transitions hists", show_deltas=False, show_weight_augmentations=show_weight_augmentations)

    sudden_dists = plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=23, x_lim_used_bias=11, x_lim_used_augment=40, bin_sets=bin_sets, title="weight changes sudden at transitions delta hists", show_deltas=True, show_weight_augmentations=show_weight_augmentations)

    average, counter, all_datapoints = plot_traces_and_collate_data(data=all_sudden_changes, augmented_data=aug_all_sudden_changes, title="sudden_weight_change")

    plot_compact(average, counter, title="compact sudden_weight_change", show_weight_augmentations=show_weight_augmentations)

    plot_compact_split(all_datapoints, title="compact sudden_weight_change", show_weight_augmentations=show_weight_augmentations, width_divisor=65)

    plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=56, x_lim_used_bias=28, x_lim_used_augment=140, bin_sets=bin_sets, title="weight changes sudden hists", show_deltas=False, show_weight_augmentations=show_weight_augmentations)

    plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=120, x_lim_used_bias=40, x_lim_used_augment=180, bin_sets=bin_sets, title="weight changes sudden delta hists", show_deltas=True, show_weight_augmentations=show_weight_augmentations)

    dur_lims = [(56, 28), (48, 25), (35, 20), (35, 17), (30, 15), (25, 12), (20, 10), (18, 9)]
    for min_dur_counter, min_dur in enumerate([2, 3, 4, 5, 7, 9, 11, 15]):
        average = np.zeros((n_weights + 1, n_types, 2))
        counter = np.zeros((n_weights + 1, n_types))
        all_datapoints = create_nested_list([n_weights + 1, n_types, 2])

        f, axs = plt.subplots(n_weights + 1, n_types, figsize=(12, 9)) # add another space to split bias into stating left versus starting right
        for weight_traj in all_weight_trajectories:

            if len(weight_traj) < min_dur or len(weight_traj) > 15: # take out too short trajectories, and too long ones
                continue

            state_type = pmf_type(weights_to_pmf(weight_traj[0]))

            for i in range(n_weights):
                if i == n_weights - 1:
                    axs[i + (weight_traj[0][i] > 0), state_type].plot([0, 1], [weight_traj[0][i], weight_traj[-1][i]], marker="o") # plot how the weight evolves from first to last appearance
                    average[i + (weight_traj[0][i] > 0), state_type] += np.array([weight_traj[0][i], weight_traj[-1][i]])
                    counter[i + (weight_traj[0][i] > 0), state_type] += 1
                    all_datapoints[i + (weight_traj[0][i] > 0)][state_type][0].append(weight_traj[0][i])
                    all_datapoints[i + (weight_traj[0][i] > 0)][state_type][1].append(weight_traj[-1][i])
                else:
                    axs[i, state_type].plot([0, 1], [weight_traj[0][i], weight_traj[-1][i]], marker="o") # plot how the weight evolves from first to last appearance
                    average[i, state_type] += np.array([weight_traj[0][i], weight_traj[-1][i]])
                    counter[i, state_type] += 1
                    all_datapoints[i][state_type][0].append(weight_traj[0][i])
                    all_datapoints[i][state_type][1].append(weight_traj[-1][i])
        for i in range(3):
            for j in range(n_weights + 1):
                axs[j, i].set_ylim(-9, 9)
                axs[j, i].spines['top'].set_visible(False)
                axs[j, i].spines['right'].set_visible(False)
                axs[j, i].set_xticks([])
                if i == 0:
                    axs[j, i].set_ylabel(ylabels[j])
                else:
                    axs[j, i].yaxis.set_ticklabels([])
                if j == 0:
                    axs[j, i].set_title("Type {}".format(i + 1))
                if j == n_weights:
                    axs[j, i].set_xlabel("Lifetime weight change")

        plt.tight_layout()
        plt.savefig("./summary_figures/weight_changes/all weight changes min dur {}".format(min_dur))
        plt.close()

        plot_compact(average, counter, title="compact changes min dur {}".format(min_dur), show_first_and_last=True)

        # means all split up
        f, axs = plt.subplots(n_weights + 1, 3, figsize=(12, 9))
        for i in range(3):
            for j in range(n_weights + 1):
                axs[j, i].plot([0, 1], average[j, i] / counter[j, i], marker="o")
                if j < 2:
                    axs[j, i].set_ylim(-4.5, 4.5)
                else:
                    axs[j, i].set_ylim(-2, 2)
                axs[j, i].spines['top'].set_visible(False)
                axs[j, i].spines['right'].set_visible(False)
                axs[j, i].set_xticks([])
                if i == 0:
                    axs[j, i].set_ylabel(ylabels[j])
                    if j < n_weights - 1:
                        axs[j, i].plot([0], [np.mean(first_and_last_pmf[:, 0, j])], marker='*', c='red')  # also plot weights of very first state average
                    if j == n_weights - 1:
                        mask = first_and_last_pmf[:, 0, -1] < 0
                        axs[j, i].plot([0], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', c='red')  # separete biases again
                    if j == n_weights:
                        mask = first_and_last_pmf[:, 0, -1] > 0
                        axs[j, i].plot([0], [np.mean(first_and_last_pmf[mask, 0, -1])], marker='*', c='red')
                else:
                    axs[j, i].yaxis.set_ticklabels([])
                if i == 2:
                    if j < n_weights - 1:
                        axs[j, i].plot([1], [np.mean(first_and_last_pmf[:, 1, j])], marker='*', c='red')  # also plot weights of very last state average
                    if j == n_weights - 1:
                        mask = first_and_last_pmf[:, 1, -1] < 0
                        axs[j, i].plot([1], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', c='red')  # separete biases again
                    if j == n_weights:
                        mask = first_and_last_pmf[:, 1, -1] > 0
                        axs[j, i].plot([1], [np.mean(first_and_last_pmf[mask, 1, -1])], marker='*', c='red')
                if j == 0:
                    axs[j, i].set_title("Type {}".format(i + 1))
                if j == n_weights:
                    axs[j, i].set_xlabel("Lifetime weight change")
        plt.savefig("./summary_figures/weight_changes/weight changes min dur {}".format(min_dur))
        plt.close()

        x_lim_used_normal, x_lim_used_bias = dur_lims[min_dur_counter]
        plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=x_lim_used_normal, x_lim_used_bias=x_lim_used_bias, bin_sets=bin_sets,
                             title="weight changes min dur {} hists".format(min_dur), show_deltas=False, show_first_and_last=True)

        if min_dur == 5:
            x_lim_used_normal, x_lim_used_bias = 25, 8
        x_lim_used_normal, x_lim_used_bias = int(x_lim_used_normal * 2.5), int(x_lim_used_bias * 2.5)
        slow_dists = plot_histogram_diffs(all_datapoints, average, counter, x_lim_used_normal=x_lim_used_normal, x_lim_used_bias=x_lim_used_bias, bin_sets=bin_sets,
                             title="weight changes min dur {} delta hists".format(min_dur), show_deltas=True, show_first_and_last=True)
        if min_dur == 5:
            import scipy
            from itertools import product
            sig_counter, insig_counter = 0, 0
            p_vals = []
            for i, j, k in product([3, 4], [0, 1], [0, 1, 2]):  # for weights 3 and 4, we compare over all combos of transitions
                print()
                print(j, i, k, i)
                print(np.mean(np.abs(sudden_dists[(j, i)])), np.mean(np.abs(slow_dists[(k, i)])))
                print(f"effect size {(np.mean(np.abs(sudden_dists[(j, i)])) - np.mean(np.abs(slow_dists[(k, i)]))) / np.std(np.abs(sudden_dists[(j, i)]))}")
                res = scipy.stats.mannwhitneyu(np.abs(sudden_dists[(j, i)]), np.abs(slow_dists[(k, i)]), alternative='greater')
                print(res)
                p_vals.append(res.pvalue)
                if res.pvalue < 0.05:
                    sig_counter += 1
                else:
                    insig_counter += 1
            print(sig_counter, insig_counter)
            print(p_vals)
            # print(scipy.stats.false_discovery_control(p_vals))  # correction does not work in this environment, copied over
            quit()

# all pmf weights

only_first_states = True

if only_first_states:
    all_first_pmfs_typeless = pickle.load(open("all_first_pmfs_typeless.p", 'rb'))
    first_pmfs_list = []
    for subject in all_first_pmfs_typeless.keys():
        for pmf in all_first_pmfs_typeless[subject]:
            first_pmfs_list.append(pmf[1])
    first_pmfs = np.array(first_pmfs_list)
else:
    apw = np.array(pickle.load(open("all_pmf_weights.p", 'rb')))

    colors = [type2color[pmf_type(weights_to_pmf(x))] for x in apw]
    colors_rew = [type2color[pmf_type_rew(x)] for x in apw]


type1_rews = []
type2_rews = []
type3_rews = []
all_rews = []

if only_first_states:
    for pmfs in first_pmfs:
        type = pmf_type(pmfs)
        all_rews.append(pmf_to_perf(pmfs))
        if type == 0:
            if pmf_to_perf(pmfs) > 0.8:
                plt.plot(pmfs)
                plt.show()
            type1_rews.append(pmf_to_perf(pmfs))
        elif type == 1:
            type2_rews.append(pmf_to_perf(pmfs))
        elif type == 2:
            type3_rews.append(pmf_to_perf(pmfs))
else:
    for weights in apw:
        type = pmf_type(weights_to_pmf(weights))
        all_rews.append(pmf_to_perf(weights_to_pmf(weights)))
        if type == 0:
            if pmf_to_perf(weights_to_pmf(weights)) > 0.8:
                plt.plot(weights_to_pmf(weights))
                plt.show()
            type1_rews.append(pmf_to_perf(weights_to_pmf(weights)))
        elif type == 1:
            type2_rews.append(pmf_to_perf(weights_to_pmf(weights)))
        elif type == 2:
            type3_rews.append(pmf_to_perf(weights_to_pmf(weights)))

type1_rews, type2_rews, type3_rews = np.array(type1_rews), np.array(type2_rews), np.array(type3_rews)

# reward rate and boundaries, figure 12
fig = plt.figure(figsize=(13 * 3 / 5, 9 * 3 / 5))
plt.hist(all_rews, bins=45, color='grey')
plt.axvline(0.6, c='k')
plt.axvline(0.7827, c='k')

plt.ylabel("# of occurences", size=28)
plt.xlabel("Reward rate", size=28)
plt.gca().spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(folder + "single hist" + " only first states" * only_first_states)
plt.close()
