"""
    This creates figure 4, and a bonus figure about how frequent which state type is during interpolated training time.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


type2color = {0: 'green', 1: 'blue', 2: 'red'}
all_conts = np.array([-1, -0.5, -.25, -.125, -.062, 0, .062, .125, .25, 0.5, 1])

performance_points = np.array([-1, -1, 0, 0])
np.random.seed(2645)  # seed for reliable samples

def pmf_to_perf(pmf):
    # determine performance of a pmf, but only on the omnipresent strongest contrasts
    return np.mean(np.abs(performance_points + pmf[[0, 1, -2, -1]]))

def pmf_type(pmf):
    rew = pmf_to_perf(pmf)
    if rew < 0.6:
        return 0
    elif rew < 0.7827:
        return 1
    else:
        return 2


if __name__ == "__main__":

    state_types_interpolation = pickle.load(open("state_types_interpolation.p", 'rb'))
    state_types_interpolation = state_types_interpolation / state_types_interpolation.max() * 100

    fs = 18

    plt.plot(np.linspace(0, 1, 150), state_types_interpolation[0], color=type2color[0])
    plt.plot(np.linspace(0, 1, 150), state_types_interpolation[1], color=type2color[1])
    plt.plot(np.linspace(0, 1, 150), state_types_interpolation[2], color=type2color[2])
    plt.ylabel("% of type across population", size=fs)
    plt.xlabel("Interpolated training time", size=fs)
    plt.ylim(0, 100)
    plt.xlim(0, 1)
    sns.despine()
    plt.tight_layout()
    plt.savefig("./summary_figures/state evos")
    plt.show()

    all_first_pmfs_typeless = pickle.load(open("all_first_pmfs_typeless.p", 'rb'))
    type_2_counter = 0
    for subject in all_first_pmfs_typeless.keys():
        type_2_intro_present = False
        for defined_points, pmf in all_first_pmfs_typeless[subject]:
            type_2_intro_present = type_2_intro_present or (pmf_type(pmf) == 1)
        type_2_counter += type_2_intro_present
    print("Out of {} mice, {} have a type 2 intro".format(len(all_first_pmfs_typeless), type_2_counter))
    # Out of 119 mice, 101 have a type 2 intro
    # Out of 134 mice, 114 have a type 2 intro

    # All first PMFs
    # I had an issue where 0 contrasts were undefined for all PMFs, check that that's not the case
    at_least_once = False
    for key in all_first_pmfs_typeless:
        for pmf in all_first_pmfs_typeless[key]:
            def_points, _ = pmf
            if def_points[5]:
                at_least_once = True
    assert at_least_once

    type_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    type_saves = [[], [], [], [], []]
    for key in all_first_pmfs_typeless:
        for pmf in all_first_pmfs_typeless[key]:

            defined_points, pmf = pmf
            temp_type = pmf_type(pmf)

            if temp_type == 0:
                defined_points = np.array([ True,  True, False, False, False, False, False, False, False, True,  True])
            elif temp_type == 1:
                defined_points = np.array([ True,  True, True, False, False, False, False, False, True, True,  True])
            else:
                defined_points[:] = True

            if temp_type == 0:
                type_saves[temp_type].append((defined_points, pmf))
            elif temp_type == 1:
                if np.abs(pmf[0] + pmf[-1] - 1) <= 0.1:
                    type_saves[3].append((defined_points, pmf))
                else:
                    type_saves[1 + int(pmf[0] > 1 - pmf[-1])].append((defined_points, pmf))
            else:
                type_saves[4].append((defined_points, pmf))

    tick_size = 14
    label_size = 34
    all_first_pmfs = pickle.load(open("all_first_pmfs.p", 'rb'))
    n_rows, n_cols = 1, 5
    fg, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6.6))
    save_title = "all types" if True else "KS014 types"

    if save_title == "KS014 types":
        all_first_pmfs_typeless = {'KS014': all_first_pmfs_typeless['KS014']}

    for i, type_save in enumerate(type_saves):
        type_array = np.empty((len(type_save), 11))
        type_array[:] = np.nan
        for j, pmf in enumerate(type_save):
            def_points, pmf_values = pmf
            type_array[j][def_points] = pmf_values[def_points]

        if i == 0:
            x = [0, 1, 9, 10]
        elif i == 4:
            x = np.arange(11)
        else:
            x = [0, 1, 2, 8, 9, 10]
        percentiles = np.percentile(type_array, [2.5, 97.5], axis=0)
        type_max = type_array.shape[0]
        sample_js = np.random.choice(np.arange(type_max), 10 if i == 0 else 5)
        for j in sample_js:
            axs[i].plot(x, type_array[j, x], c=type2color[type_map[i]], alpha=0.45)
        axs[i].plot(x, np.mean(type_array[:, x], axis=0), c=type2color[type_map[i]], lw=5)
        axs[i].fill_between(x, percentiles[1, x], percentiles[0, x], alpha=0.2, color=type2color[type_map[i]])
        axs[i].annotate("N={}".format(len(type_save)), (5.75, 0.035), size=22)

        axs[i].set_ylim(0, 1)
        axs[i].set_xticks(np.arange(11), ['-1', '', '-0.25', '', '', '0', '', '', '0.25', '', '1'], size=tick_size, rotation=45)
        if i == 0:
            axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], size=tick_size)
        else:
            axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1], ['']*5, size=tick_size)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xlim(0, 10)
    axs[0].set_ylabel("P(rightwards)", size=label_size)
    axs[2].set_xlabel("Contrasts", size=label_size)

    offset_x = 0.25
    axs[0].annotate("a", (offset_x, 1), weight='bold', fontsize=22)
    axs[1].annotate("b", (offset_x, 1), weight='bold', fontsize=22)
    axs[2].annotate("c", (offset_x, 1), weight='bold', fontsize=22)
    axs[3].annotate("d", (offset_x, 1), weight='bold', fontsize=22)
    axs[4].annotate("e", (offset_x, 1), weight='bold', fontsize=22)

    axs[0].set_title("Type 1", size=label_size-2, color='green')
    axs[2].set_title("Type 2", size=label_size-2, color='blue')
    axs[4].set_title("Type 3", size=label_size-2, color='red')

    plt.tight_layout()
    plt.savefig("./summary_figures/" + save_title, dpi=300)
    plt.show()