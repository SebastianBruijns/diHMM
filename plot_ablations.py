import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil
import matplotlib.gridspec as gridspec
import pickle

folder = "./dynamic_GLMiHMM_crossvals/infos_ablations/"
doubles = "./dynamic_GLMiHMM_crossvals/infos_extra/"

redo = False

# Define conditions

condition_names = ["Best model",
                   "3 states, no slow proc.",
                   "15 states, no slow proc.",
                   "1 state",
                   "Best + WSLS",
                   "No perseveration",
                   "No durations (exp. only)",]

# no-shows: 'NYU-21', 'ZFM-04308' 
subject_names = ['CSHL045', 'CSHL047', 'CSHL049', 'CSHL051', 'CSHL052', 'CSHL053', 'CSHL054', 'CSHL055', 'CSHL058', 'CSHL059', 'CSHL060', 'CSHL_007', 'CSHL_014', 'CSHL_015',
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

subjects_and_nums = [(subject, num) for subject in subject_names for num in range(2)]

condition_subs = {i: subjects_and_nums.copy() for i in range(len(condition_names))}

conditions = [
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' in x and x['dur'] == 'no') and x['n_states'] == 1,
    # lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' in x and x['dur'] == 'no') and x['n_states'] == 1,
    lambda x: '/summarised_sessions/0_25_wsls/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 3 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0. and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' in x and x['dur'] == 'yes') and x['n_states'] == 15 and x['dur_params']['beta_0'] == 500,
    lambda x: '/summarised_sessions/0_25_no_exp/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100
]


if redo:
    # Initialize dictionaries to store results
    cvlls = {i: [] for i in range(len(conditions))}
    subjects = {i: [] for i in range(len(conditions))}
    best_files = []
    no_slow_15 = []

    for fol in [folder, doubles]:
        for file in tqdm(os.listdir(fol)):

            infos = json.load(open(fol + file, 'r'))
            subject = infos['subject']
            if not infos['cross_val']:
                continue
            if infos['n_samples'] == 10:
                continue
            assert infos['n_samples'] in [12000, 10000]
            # if subject == "KS014":
            #     print(file)

            if infos['cross_val_num'] not in [0, 1]:
                continue

            for i, condition in enumerate(conditions):
                if conditions[i](infos):
                    cvlls[i].append(np.mean(infos['cross_val_preds'][-4000:]))
                    subjects[i].append(subject)
                    condition_subs[i].remove((subject, infos['cross_val_num']))
                    # os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_ablations/" + file)

                    if i == 0:
                        # file looks like this [stuff]_[number].json, using a regular expression, extract the number
                        number = int(file.split('_')[-1].split('.')[0])
                        best_files.append(f"{subject}_crossval_{infos['cross_val_num']}_0.04_var_{infos['seed']}_{number}_2.p")
                    elif i == 5:
                        number = int(file.split('_')[-1].split('.')[0])
                        no_slow_15.append(f"{subject}_crossval_{infos['cross_val_num']}_0.0_var_{infos['seed']}_{number}_2.p")
                    # if i == 0:
                    #     shutil.copyfile(fol + file, "./dynamic_GLMiHMM_crossvals/infos/" + file)
                    #     shutil.copyfile(fol + file, "./dynamic_GLMiHMM_crossvals/infos_model_comparisons/" + file)
                    break
    pickle.dump(cvlls, open("cvlls_ablat.pkl", 'wb'))
    pickle.dump(subjects, open("subjects_ablat.pkl", 'wb'))
else:
    cvlls = pickle.load(open("cvlls_ablat.pkl", 'rb'))
    subjects = pickle.load(open("subjects_ablat.pkl", 'rb'))


fig = plt.figure(figsize=(14, 6))
spec = gridspec.GridSpec(ncols=100, nrows=100, figure=fig)
spec.update(hspace=0.)  # set the spacing between axes.
ax0 = fig.add_subplot(spec[:, :60])  # accuracy trace
ax1 = fig.add_subplot(spec[:, 70:])

from scipy.stats import sem

print([len(cvlls[x]) for x in cvlls])

# plot lines of the different means
vals = []
for i, label in enumerate(condition_names):
    ax1.errorbar(i, np.mean(cvlls[i]), yerr=sem(cvlls[i]) / 2, fmt='o', label=label)
    if i == 0:
        plt.axhline(np.mean(cvlls[i]), color='k', alpha=0.25)
    vals.append(np.mean(cvlls[i]))
    print(i, label, np.mean(cvlls[i]))
# ax1.set_ylabel("Trialwise log-likelihood", fontsize=16)
ax1.set_xticks(range(len(conditions)), condition_names, rotation=45, fontsize=12, ha='right')
ax1.set_ylim(np.mean(cvlls[0]) - 4 * 0.018, np.mean(cvlls[0]) + 1.25 * 0.018)
# plt.ylim()
# plt.savefig("means.png")

redo = False
folder = "./dynamic_GLMiHMM_crossvals/infos_model_comparisons/"
# doubles = "./dynamic_GLMiHMM_crossvals/infos/"

condition_names = ["Best model",
                   "Pers. decay =0.15",
                   "Pers. decay =0.2",
                   "Pers. decay =0.3",
                   "Slow variance =0.03",
                   "Slow variance =0.06",
                   "Duration higher r",
                   "Weak limit L=18",
                   r"$\gamma$ prior =(0.1, 0.1)",
                   r"$\gamma$ prior =(0.001, 0.001)",
                   r"$\alpha$ prior =(0.1, 0.1)",
                   r"$\alpha$ prior =(0.001, 0.001)",
]

conditions = [
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_15/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_2/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_3/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.03 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.06 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 905)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 18 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.1 and x['gamma_b_0'] == 10 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.001 and x['gamma_b_0'] == 1000 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.1 and x['alpha_b_0'] == 10,
    lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and len(x['dur_params']['r_support']) == 175 and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.001 and x['alpha_b_0'] == 1000,
]


if redo:
    cvlls = {i: [] for i in range(len(conditions))}
    subjects = {i: [] for i in range(len(conditions))}

    for fol in [folder, doubles]:
        for file in tqdm(os.listdir(fol)):

            infos = json.load(open(fol + file, 'r'))
            subject = infos['subject']
            if not infos['cross_val']:
                continue
            if infos['n_samples'] == 10:
                continue
            assert infos['n_samples'] in [12000, 10000]
            # if subject == "KS014":
            #     print(file)

            if infos['cross_val_num'] not in [0, 1]:
                continue

            for i, condition in enumerate(conditions):
                if conditions[i](infos):
                    if subject in subjects[i] and np.mean(infos['cross_val_preds'][-4000:]) in cvlls[i]:
                        # print(file)
                        os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
                        break
                    cvlls[i].append(np.mean(infos['cross_val_preds'][-4000:]))
                    subjects[i].append(subject)
                    # os.rename(fol + file, "./dynamic_GLMiHMM_crossvals/infos_model_comparisons/" + file)
                    break
    pickle.dump(cvlls, open("cvlls.pkl", 'wb'))
    pickle.dump(subjects, open("subjects.pkl", 'wb'))
else:
    cvlls = pickle.load(open("cvlls.pkl", 'rb'))
    subjects = pickle.load(open("subjects.pkl", 'rb'))


print([len(cvlls[x]) for x in cvlls])

for i, label in enumerate(condition_names):
    ax0.errorbar(i, np.mean(cvlls[i]), yerr=sem(cvlls[i]) / 2, fmt='o', label=label)
    if i == 0:
        ax0.axhline(np.mean(cvlls[i]), color='k', alpha=0.25)
    vals.append(np.mean(cvlls[i]))
    print(i, label, np.mean(cvlls[i]))
ax0.set_ylabel("Trialwise LL on heldout", fontsize=22)
ax0.set_xticks(range(len(conditions)), condition_names, rotation=45, fontsize=12, ha='right')
ax0.set_ylim(np.mean(cvlls[0]) - 4 * 0.0012, np.mean(cvlls[0]) + 1.25 * 0.0012)


from matplotlib.patches import ConnectionPatch
con1 = ConnectionPatch(xyA=(len(conditions)-0.45, np.mean(cvlls[0]) + 1.25 * 0.0012), 
                       coordsA=ax0.transData, 
                       xyB=(-0.3, np.mean(cvlls[0]) + 1.25 * 0.0012), 
                       coordsB=ax1.transData, 
                       axesA=ax0, 
                       axesB=ax1, 
                       arrowstyle="-", 
                       color='k', 
                       linewidth=1, alpha=0.5)
fig.add_artist(con1)

# Bottom connection
con2 = ConnectionPatch(xyA=(len(conditions)-0.45, np.mean(cvlls[0]) - 4 * 0.0012), 
                       coordsA=ax0.transData,
                       xyB=(-0.3, np.mean(cvlls[0]) - 4 * 0.0012),
                       coordsB=ax1.transData,
                       axesA=ax0,
                       axesB=ax1,
                       arrowstyle="-",
                       color='k',
                       linewidth=1, alpha=0.5)
fig.add_artist(con2)


plt.tight_layout(pad=7.0)
plt.savefig("model_comp_n_ablations", dpi=200, bbox_inches='tight')
plt.show()