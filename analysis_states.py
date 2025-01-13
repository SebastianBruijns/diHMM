"""
    Unused figures on the connection between regressions and passed time to last session.
    TODO: check wether dates are correct
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

regressed_or_not_list = pickle.load(open("multi_chain_saves/regressed_or_not_list.p", 'rb'))
regression_magnitude_list = pickle.load(open("multi_chain_saves/regression_magnitude_list.p", 'rb'))
dates_list = pickle.load(open("multi_chain_saves/dates_list.p", 'rb'))

i, j = 0, 0
dates_fixed = []
regressed_or_not_list_fixed = []
regression_magnitude_fixed = []
while True:
    if i >= len(dates_list) or j >= len(regressed_or_not_list):
        break
    if len(dates_list[i]) - 1 == len(regressed_or_not_list[j]):
        dates_fixed.append(dates_list[i])
        regression_magnitude_fixed.append(regression_magnitude_list[i])
        regressed_or_not_list_fixed.append(regressed_or_not_list[i])
        i += 1
        j += 1
    else:
        i += 1
        j += 1
        print(False, i - 1)

regressed_or_not_list = regressed_or_not_list_fixed
regression_magnitude_list = regression_magnitude_fixed

dates_diff = []
for sub_dates in dates_fixed:
    temp = []
    for i in range(len(sub_dates) - 1):
        temp.append(sub_dates[i+1] - sub_dates[i])
    dates_diff.append(temp)

regressed_or_not_list = [item for sublist in regressed_or_not_list for item in sublist]
regression_magnitude_list = [item for sublist in regression_magnitude_list for item in sublist]
dates_diff = [item for sublist in dates_diff for item in sublist]

plt.scatter([x.total_seconds() for x in dates_diff], regressed_or_not_list)
plt.show()

plt.scatter([x.total_seconds() for x in dates_diff], regression_magnitude_list)
plt.show()

from scipy.stats import pearsonr, mannwhitneyu
print(pearsonr([x.total_seconds() for x in dates_diff], np.array(regression_magnitude_list)))
a = np.array(regressed_or_not_list)
b = np.array([x.total_seconds() for x in dates_diff])
print(mannwhitneyu(b[a == 0], b[a == 1]))


captured_states = pickle.load(open("captured_states.p", 'rb'))

# captured states is: captured_states.append((len([item for sublist in state_sets for item in sublist if len(sublist) > 40]), test.results[0].n_datapoints, len([s for s in state_sets if len(s) > 40])))
num_trials = np.array([x for _, x, _, _ in captured_states])
num_covered_trials = np.array([x for x, _, _, _ in captured_states])
num_states = np.array([x for _, _, x, _ in captured_states])
num_sessions = np.array([x for _, _, _, x in captured_states])

# print(num_covered_trials / num_trials)
print("Mean fraction of accounted trials: {}".format(np.mean(num_covered_trials / num_trials)))
print("Minimum fraction of accounted trials: {}".format(np.min(num_covered_trials / num_trials)))

print(np.unique(num_states, return_counts=True))


type_1_to_2_save = pickle.load(open("multi_chain_saves/type_1_to_2_save.p", 'rb'))

previously_expressed = 0
not_expressed = 0
counter = 0
all_prev_biases = []
all_biases = []
neutral_counter, symm_counter = 0, 0
boring_type_2 = 0
for i, pmf_lists in enumerate(type_1_to_2_save):
    if pmf_lists == [[], []]:
        continue
    counter += 1
    expressed_biases = []
    for type_1_pmf in pmf_lists[0]:
        if np.mean(type_1_pmf[[0, 1, -2, -1]]) < 0.45:
            if -1 not in expressed_biases:
                expressed_biases.append(-1)
        elif np.mean(type_1_pmf[[0, 1, -2, -1]]) > 0.55:
            if 1 not in expressed_biases:
                expressed_biases.append(1)
        else:
            if 0 not in expressed_biases:
                expressed_biases.append(0)
    if np.mean(pmf_lists[1][0][[0, 1, -2, -1]]) < 0.45:
        type_2_bias = -1
    elif np.mean(pmf_lists[1][0][[0, 1, -2, -1]]) > 0.55:
        type_2_bias = 1
    else:
        type_2_bias = 0

    all_prev_biases.append(expressed_biases)
    all_biases.append(expressed_biases + [type_2_bias])

    if type_2_bias == 0:
         boring_type_2 += 1

    if type_2_bias in expressed_biases:
        previously_expressed += 1

    else:
        not_expressed += 1


print(boring_type_2)
print(previously_expressed, not_expressed)

from scipy.stats import binomtest
print(binomtest(previously_expressed, previously_expressed + not_expressed, 0.5))

from scipy.stats import linregress

quantiles = np.linspace(0, 1, 2)[1:]

quant_sessions = np.quantile(num_trials, quantiles)
prev_session_bound = num_trials.min() - 1

for quant_session in quant_sessions:
    mask = np.logical_and(prev_session_bound < num_trials, num_trials <= quant_session)
    print("Num of mice considered {}".format(np.sum(mask)))

    res = linregress(num_trials[mask], num_states[mask])
    plt.plot([prev_session_bound, quant_session], [res.intercept + res.slope * prev_session_bound, res.intercept + res.slope * quant_session])
    prev_session_bound = quant_session

plt.scatter(num_trials, num_states)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("./summary_figures/state_num_regression.png", dpi=300)
plt.show()



session_time_at_sudden_changes = pickle.load(open("multi_chain_saves/session_time_at_sudden_changes.p", 'rb'))

f, axs = plt.subplots(2, 1, figsize=(16 * 0.75, 9 * 0.75), sharex=True, sharey=True)
axs[0].hist(session_time_at_sudden_changes[0])
axs[1].hist(session_time_at_sudden_changes[1])
axs[0].set_ylabel("First type 2 intro")
axs[1].set_ylabel("First type 3 intro")
axs[1].set_xlabel("Interpolated session time")
plt.tight_layout()
plt.savefig("./summary_figures/new type intro points.png", dpi=300)
plt.show()