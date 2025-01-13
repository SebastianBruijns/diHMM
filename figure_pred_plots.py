import numpy as np
import matplotlib.pyplot as plt
from dyn_glm_chain_analysis import predictive_check
import pickle
import matplotlib.gridspec as gridspec
import os

subject = "KS014"
fit_type = 'prebias'
folder = "./pred_checks_5/"
fit_variance = 0.04

acc_data, cont_data = [(3, 5), (6, 8), (0, 2)][1] # 5 is minus last 50 trials

bin_num = 20
fs = 16

# test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}".format(subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# mode_indices = pickle.load(open("multi_chain_saves/{}_mode_indices_{}_{}".format('first', subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# true_accs, gen_accs, true_pmf, gen_pmf = predictive_check(test, mode_indices, return_acc_plot=True)
# pickle.dump((true_accs, gen_accs, true_pmf, gen_pmf), open("temp_pred_plots", 'wb'))
true_accs, gen_accs, true_pmf, gen_pmf = pickle.load(open("temp_pred_plots", 'rb'))

fig = plt.figure(figsize=(15, 8))
spec = gridspec.GridSpec(ncols=100, nrows=100, figure=fig)
spec.update(hspace=0.)  # set the spacing between axes.
ax0 = fig.add_subplot(spec[:45, :20])  # accuracy trace
ax1 = fig.add_subplot(spec[:45, 27:44])
ax2 = fig.add_subplot(spec[:45, 51:75])
ax3 = fig.add_subplot(spec[:45, 82:])

ax4 = fig.add_subplot(spec[58:, :17])
ax4_add = fig.add_subplot(spec[59:81, 7:17])
ax5 = fig.add_subplot(spec[58:, 22:38])
ax6 = fig.add_subplot(spec[58:, 42:58])
ax7 = fig.add_subplot(spec[58:, 63:79])
ax8 = fig.add_subplot(spec[58:, 83:])
ax8_add = fig.add_subplot(spec[58:78, 87:96])

offset_x, offset_y = 0.025, 1
ax0.annotate("a", (offset_x, offset_y), weight='bold', fontsize=22, xycoords='axes fraction')
ax1.annotate("b", (offset_x, offset_y), weight='bold', fontsize=22, xycoords='axes fraction')
ax2.annotate("c", (offset_x, offset_y), weight='bold', fontsize=22, xycoords='axes fraction')
ax3.annotate("d", (offset_x, offset_y), weight='bold', fontsize=22, xycoords='axes fraction')
ax4.annotate("e", (offset_x, offset_y), weight='bold', fontsize=22, xycoords='axes fraction')


gen_mean = []
percentiles_50 = []
percentiles_95 = []

for ga in gen_accs:
    gen_mean.append(np.mean(ga))
    percentiles_50.append([np.percentile(ga, 25), np.percentile(ga, 75)])
    percentiles_95.append([np.percentile(ga, 2.5), np.percentile(ga, 97.5)])


ax0.plot(range(1, 1 + len(true_accs)), true_accs, 'k', zorder=3, label='Empirical')
ax0.plot(range(1, 1 + len(true_accs)), gen_mean, 'g', zorder=2, label='Gen. mean')
ax0.fill_between(range(1, 1 + len(true_accs)), np.array(percentiles_50)[:, 0], np.array(percentiles_50)[:, 1], color='b', zorder=1, alpha=0.25, label='Gen. 50% CI')
ax0.fill_between(range(1, 1 + len(true_accs)), np.array(percentiles_95)[:, 0], np.array(percentiles_95)[:, 1], color='r', zorder=0, alpha=0.1, label='Gen. 95% CI')

ax0.set_xlim(1, len(true_accs))
ax0.legend(frameon=False)
ax0.set_ylabel("% correct", fontsize=fs)
ax0.set_xlabel("Session", fontsize=fs)
ax0.spines[['right', 'top']].set_visible(False)



ax1.plot(true_pmf, 'k', zorder=3, label='Empirical')
ax1.plot(np.mean(gen_pmf, axis=0), 'g', zorder=2, label='Gen. mean')
percentiles = np.percentile(gen_pmf, [25, 75], axis=0)
ax1.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='b', zorder=1, alpha=0.25, label='Gen. 50% CI')
percentiles = np.percentile(gen_pmf, [2.5, 97.5], axis=0)
ax1.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='r', zorder=0, alpha=0.1, label='Gen. 95% CI')
ax1.set_xlim(0, len(true_pmf) - 1)
ax1.set_xticks([0, 4, 8], [-1, 0, 1])
ax1.spines[['right', 'top']].set_visible(False)
ax1.tick_params(axis='both', which='major', labelsize=fs-5)
ax1.set_xlabel("Contrast", size=fs)
ax1.set_ylabel("% rightwards responses", size=fs)
ax1.set_ylim(0, 1)



state_assistance = False
strict_smaller = False

percentiles = []
counter = 0
for file in os.listdir(folder):
    if file.startswith('all_infos_'):
        data = pickle.load(open(folder + file, 'rb'))

        if ('state_ass_True' in file) != state_assistance:
            continue

        if ('strict_smaller' in file) != strict_smaller:
            continue

        if "ibl_witten_27" in file:  # somehow only 1 cont in first session???
            continue

        # accuracy
        percentiles += data[acc_data]
        counter += 1

print(f"counter of animals is {counter}")
ax2.hist(percentiles, bins=np.linspace(0, 1, bin_num), color='grey')
ax2.set_xlabel("True acc. percentile", size=fs)
ax2.set_ylabel("# of sessions", size=fs)
ax2.spines[['right', 'top']].set_visible(False)
ax2.set_xlim(0, 1)

percentiles = []
counter = 0
for file in os.listdir(folder):
    if file.startswith('all_infos_'):
        data = pickle.load(open(folder + file, 'rb'))

        if ('state_ass_True' in file) != state_assistance:
            continue

        if ('strict_smaller' in file) != strict_smaller:
            continue

        
        if "ibl_witten_27" in file:  # somehow only 1 cont in first session???
            continue

        # all contrasts
        percentiles += [item for sublist in data[cont_data] for item in sublist]
        counter += 1

print(f"counter of animals is {counter}")
ax3.hist(percentiles, bins=np.linspace(0, 1, bin_num), color='grey')
ax3.set_xlabel("True choice percentile", size=fs)
ax3.set_ylabel("# of sessions and contrasts", size=fs)
ax3.spines[['right', 'top']].set_visible(False)
ax3.set_xlim(0, 1)

cont_names = {0: "Contrast -1", 1: "Contrast -0.5", -2: "Contrast 0.5", -1: "Contrast 1"}
for cont_of_int, ax in zip([0, 1, -2, -1], [ax4, ax5, ax7, ax8]):
    percentiles = []
    counter = 0
    for file in os.listdir(folder):
        if file.startswith('all_infos_'):
            data = pickle.load(open(folder + file, 'rb'))

            if ('state_ass_True' in file) != state_assistance:
                continue

            if ('strict_smaller' in file) != strict_smaller:
                continue

            
            if "ibl_witten_27" in file:  # somehow only 1 cont in first session???
                continue

            # all contrasts
            if cont_of_int in [0, -1]:
                percentiles += [x[cont_of_int] for x in data[cont_data]]
            else:
                percentiles += [x[cont_of_int] for x in data[cont_data] if len(x) != 9]
            counter += 1

    ax.hist(percentiles, bins=np.linspace(0, 1, bin_num), color='grey', label=cont_names[cont_of_int])
    if cont_of_int == 0:
        ax.set_xlabel("True choice percentile", size=fs)
        ax.set_ylabel("# of sessions", size=fs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_title(cont_names[cont_of_int])


# ax4 inset
subject, session = "CSHL045", 15
# test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}".format(subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# mode_indices = pickle.load(open("multi_chain_saves/{}_mode_indices_{}_{}".format('first', subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# true_accs, gen_accs, true_pmf, gen_pmf = predictive_check(test, mode_indices, return_acc_plot=True, till_session=session)
# pickle.dump((true_accs, gen_accs, true_pmf, gen_pmf), open(f"temp_pred_plots_{subject}", 'wb'))
true_accs, gen_accs, true_pmf, gen_pmf = pickle.load(open(f"temp_pred_plots_{subject}", 'rb'))

ax4_add.plot(true_pmf, 'k', zorder=3, label='Empirical')
ax4_add.plot(np.mean(gen_pmf, axis=0), 'g', zorder=2, label='Gen. mean')
percentiles = np.percentile(gen_pmf, [25, 75], axis=0)
ax4_add.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='b', zorder=1, alpha=0.25, label='Gen. 50% CI')
percentiles = np.percentile(gen_pmf, [2.5, 97.5], axis=0)
ax4_add.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='r', zorder=0, alpha=0.1, label='Gen. 95% CI')
ax4_add.set_xlim(0, 1.5)
ax4_add.set_xticks([0, 1], [-1, -0.5])
ax4_add.spines[['right', 'top']].set_visible(False)
ax4_add.tick_params(axis='both', which='major', labelsize=fs-5)
# ax4.set_xlabel("Contrast", size=fs)
# ax4.set_ylabel("% rightwards responses", size=fs)
ax4_add.set_ylim(0, 0.5)
ax4.annotate("", xytext=(0.09, 0.77), xy=(0.4, 0.77), xycoords='axes fraction', arrowprops=dict(facecolor='black', headwidth=10, headlength=10, width=2))


# ax 8 inset
subject, session = "ibl_witten_14", 5
# test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}".format(subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# mode_indices = pickle.load(open("multi_chain_saves/{}_mode_indices_{}_{}".format('first', subject, fit_type) + "_var_{}".format(fit_variance) + '.p', 'rb'))
# true_accs, gen_accs, true_pmf, gen_pmf = predictive_check(test, mode_indices, return_acc_plot=True, till_session=session)
# pickle.dump((true_accs, gen_accs, true_pmf, gen_pmf), open(f"temp_pred_plots_{subject}", 'wb'))
true_accs, gen_accs, true_pmf, gen_pmf = pickle.load(open(f"temp_pred_plots_{subject}", 'rb'))

ax8_add.plot(true_pmf, 'k', zorder=3, label='Empirical')
ax8_add.plot(np.mean(gen_pmf, axis=0), 'g', zorder=2, label='Gen. mean')
percentiles = np.percentile(gen_pmf, [25, 75], axis=0)
ax8_add.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='b', zorder=1, alpha=0.25, label='Gen. 50% CI')
percentiles = np.percentile(gen_pmf, [2.5, 97.5], axis=0)
ax8_add.fill_between(range(len(true_pmf)), percentiles[0], percentiles[1], color='r', zorder=0, alpha=0.1, label='Gen. 95% CI')
ax8_add.set_xlim(6.5, 9)
ax8_add.set_xticks([7, 8, 9], [0.25, 0.5, 1])
ax8_add.spines[['right', 'top']].set_visible(False)
ax8_add.tick_params(axis='both', which='major', labelsize=fs-5)
# ax8.set_xlabel("Contrast", size=fs)
# ax8.set_ylabel("% rightwards responses", size=fs)
ax8_add.set_ylim(0.5, 1)
ax8.annotate("", xytext=(0.93, 0.87), xy=(0.78, 0.87), xycoords='axes fraction', arrowprops=dict(facecolor='black', headwidth=10, headlength=10, width=2))



percentiles = []
counter = 0
for file in os.listdir(folder):
    if file.startswith('all_infos_'):
        data = pickle.load(open(folder + file, 'rb'))

        if ('state_ass_True' in file) != state_assistance:
            continue

        if ('strict_smaller' in file) != strict_smaller:
            continue

        
        if "ibl_witten_27" in file:  # somehow only 1 cont in first session???
            continue

        # all contrasts
        percentiles += [x[5] for x in data[cont_data] if len(x) == 11]
        percentiles += [x[4] for x in data[cont_data] if len(x) == 9]
        counter += 1

ax6.hist(percentiles, bins=np.linspace(0, 1, bin_num), color='grey', label="Cont. 0")
ax6.spines[['right', 'top']].set_visible(False)
ax6.set_xlim(0, 1)
ax6.set_title("Contrast 0")



plt.tight_layout()
plt.savefig("./summary_figures/pred_plot")
plt.show()