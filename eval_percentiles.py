import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# order of returns
# acc_percentiles, pers_percentiles, pmf_percentiles_75, acc_percentiles_50, pers_percentiles_50, pmf_percentiles_50, acc_percentiles_total, pers_percentiles_total, pmf_percentiles_total

all_relevants_a = []
all_relevants_b = []
all_relevants_c = []
all_relevants_d = []
all_relevants_weighted = []

check_a = []
check_b = []

diffs_a = []
diffs_b = []

dists_0 = []
bad_dists_0 = []

dists_acc = []
bad_dists_acc = []

bin_num = 20

contrast_of_int = 0

state_assistance = False
strict_smaller = False

title = ""

if state_assistance:
    title += "state_assistance_"

if strict_smaller:
    title += "strict_smaller_"

folder = ["./pred_checks_5/", "./pred_checks_6/", "./pred_checks_7/", "./pred_checks_8/"][2]

ass = []
bs = []
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

        if len(data) < 10:
            continue

        # extreme contrasts
        # if (np.sum(np.array([x[contrast_of_int] for x in data[8]]) < (1 / bin_num)) / len([x[contrast_of_int] for x in data[2]])) > 0.33:
        #     continue
        # all_relevants_a += [x[contrast_of_int] for x in data[2]]
        # all_relevants_b += [x[contrast_of_int] for x in data[5]]
        # all_relevants_c += [x[contrast_of_int] for x in data[8]]

        # for d, p in zip([x[contrast_of_int] for x in data[12]], [x[contrast_of_int] for x in data[8]]):
        #     if p < 0.025 or p > 0.975:
        #         bad_dists_0.append(d)
        #     else:
        #         dists_0.append(d)

        # check_a += [x[0] for x in data[8] if len(x) != 9]
        # check_b += [x[1] for x in data[8] if len(x) != 9]

        # diffs_a += [x[0] for x in data[12] if len(x) != 9]
        # diffs_b += [x[1] for x in data[12] if len(x) != 9]

        # print(len([x[contrast_of_int] for x in data[2]]), np.sum(np.array([x[contrast_of_int] for x in data[8]]) < 1 / bin_num))
        # ass.append(len([x[contrast_of_int] for x in data[2]]))
        # bs.append(np.sum(np.array([x[contrast_of_int] for x in data[8]]) < 1 / bin_num))
        # if counter == 0:
        #     title += f"_single_cont_{contrast_of_int}"
        
        # print(file, np.sum(np.array([x[0] for x in data[2]]) < 0.03), np.sum(np.array([x[-1] for x in data[2]]) > 0.98))
        # if np.sum(np.array([x[0] for x in data[2]]) < 0.02) > 5:
        #     print(np.array([x[0] for x in data[2]]))

        # 0 contrast if present
        # all_relevants_a += [x[4] for x in data[2] if len(x) == 9]
        # all_relevants_b += [x[4] for x in data[5] if len(x) == 9]
        # all_relevants_c += [x[4] for x in data[8] if len(x) == 9]
        # all_relevants_a += [x[5] for x in data[2] if len(x) == 11]
        # all_relevants_b += [x[5] for x in data[5] if len(x) == 11]
        # all_relevants_c += [x[5] for x in data[8] if len(x) == 11]

        # left 25 contrast if present
        # all_relevants_a += [x[2] for x in data[2] if len(x) >= 6 and len(x) != 9]
        # all_relevants_b += [x[2] for x in data[5] if len(x) >= 6 and len(x) != 9]
        # all_relevants_c += [x[2] for x in data[8] if len(x) >= 6 and len(x) != 9]
        # all_relevants_a += [x[1] for x in data[2] if len(x) == 9]
        # all_relevants_b += [x[1] for x in data[5] if len(x) == 9]
        # all_relevants_c += [x[1] for x in data[8] if len(x) == 9]

        # right 25 contrast if present
        # all_relevants_a += [x[-3] for x in data[2] if len(x) >= 6 and len(x) != 9]
        # all_relevants_b += [x[-3] for x in data[5] if len(x) >= 6 and len(x) != 9]
        # all_relevants_c += [x[-3] for x in data[8] if len(x) >= 6 and len(x) != 9]
        # all_relevants_a += [x[-2] for x in data[2] if len(x) == 9]
        # all_relevants_b += [x[-2] for x in data[5] if len(x) == 9]
        # all_relevants_c += [x[-2] for x in data[8] if len(x) == 9]

        # all_contrasts
        # all_relevants_a += [item for sublist in data[2] for item in sublist]  # write sublist[:-1] at end to exclude strong right contrast
        # all_relevants_b += [item for sublist in data[5] for item in sublist]
        # all_relevants_c += [item for sublist in data[8] for item in sublist]


        # accuracy
        all_relevants_a += data[0]
        all_relevants_b += data[3]
        all_relevants_c += data[6]
        for d, p in zip(data[11][0], data[6]):
            if p < 0.025 or p > 0.975:
                bad_dists_acc.append(d)
            else:
                dists_acc.append(d)
        if len(data) == 11:
            all_relevants_d += data[9]
        all_relevants_weighted += data[13]

        # perseveration
        # all_relevants_a += data[1]
        # all_relevants_b += data[4]
        # all_relevants_c += data[7]

        counter += 1
    else:
        continue


# diffs_a, diffs_b = np.array(diffs_a), np.array(diffs_b)
# print(np.sum(np.logical_and(diffs_a < 0, diffs_b < 0)))
# print(np.sum(np.logical_and(diffs_a > 0, diffs_b < 0)))
# print(np.sum(np.logical_and(diffs_a < 0, diffs_b > 0)))
# print(np.sum(np.logical_and(diffs_a > 0, diffs_b > 0)))

# plt.scatter(diffs_a, diffs_b)
# plt.xlabel("-1 diff")
# plt.ylabel("-0.5 diff")
# plt.plot([-0.4, 0.6], [-0.4, 0.6], 'k', alpha=0.3)
# plt.axhline(0, color='k', alpha=0.3)
# plt.axvline(0, color='k', alpha=0.3)
# plt.gca().set_aspect('equal')
# plt.show()


# a, b = np.array(check_a), np.array(check_b)
# points = np.linspace(0, 1, 21)

# for i in range(20):
#     plt.subplot(4, 5, i+1)
#     plt.title(f"{np.round(points[i], 2)} - {np.round(points[i+1], 2)}")
#     print(np.sum(np.logical_and(points[i] < a, points[i+1] >= a)))
#     plt.hist(b[np.logical_and(points[i] < a, points[i+1] >= a)])


# plt.figure()
# plt.scatter(a, b)
# plt.show()

# print(counter)

plt.figure()
plt.title(title.replace("_", " ") + f" n={counter}" + " 75t removed")
plt.hist(all_relevants_a, np.linspace(0, 1, bin_num))
plt.savefig(f"./pred_checks/{title}_percentiles_75_removed.png")

plt.figure()
plt.title(title.replace("_", " ") + f" n={counter}" + " 50t removed")
plt.hist(all_relevants_b, np.linspace(0, 1, bin_num))
plt.savefig(f"./pred_checks/{title}_percentiles_50_removed.png")
# plt.close()

plt.figure()
plt.title(title.replace("_", " ") + f" n={counter}" + " all trials")
plt.hist(all_relevants_c, np.linspace(0, 1, bin_num))
plt.savefig(f"./pred_checks/{title}_percentiles_all_trials.png")
# plt.close()

plt.figure()
plt.title(title.replace("_", " ") + f" n={counter}" + " weighted trials")
plt.hist(all_relevants_weighted, np.linspace(0, 1, bin_num))
plt.savefig(f"./pred_checks/{title}_percentiles_all_trials.png")

# if len(all_relevants_d) > 0:
#     plt.figure()
#     plt.title(title.replace("_", " ") + f" n={counter}" + " no full contrasts")
#     plt.hist(all_relevants_d, np.linspace(0, 1, bin_num))
#     plt.savefig(f"./pred_checks/{title}_percentiles_all_trials.png")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# plt.title(title.replace("_", " ") + f" n={counter}" + " all trials")

lims = (-0.5, 0.5) if folder == "./pred_checks_5/" else (-4, 4)
bad_hist = axs[0].hist(bad_dists_0, np.linspace(*lims, 3*bin_num), color='r', label="Outside of 95% of CI")
# stack the hist of dists on top
axs[0].hist(dists_0, np.linspace(*lims, 3*bin_num), bottom=bad_hist[0], color='g', label="Within 95% of CI")
axs[0].set_ylabel("# of sessions", fontsize=20)
axs[0].set_xlabel(r"$\Delta$PMF on -100% empirical vs posterior mean", fontsize=20)
axs[0].set_xlim(*lims)
axs[0].legend(frameon=False, fontsize=16)

lims = (-0.4, 0.4) if folder == "./pred_checks_5/" else (-4, 4)
bad_hist = axs[1].hist(bad_dists_acc, np.linspace(*lims, 3*bin_num), color='r')
# stack the hist of dists on top
axs[1].hist(dists_acc, np.linspace(*lims, 3*bin_num), bottom=bad_hist[0], color='g')
axs[1].set_xlabel(r"$\Delta$accuracy empirical vs posterior mean", fontsize=20)
axs[1].set_xlim(*lims)

sns.despine()
plt.tight_layout()
plt.savefig(f"./summary_figures/posterior_distances.png")
plt.show()