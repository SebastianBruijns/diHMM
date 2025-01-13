import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


abs_state_durs_true = pickle.load(open("multi_chain_saves/abs_state_durs.p", 'rb'))

# abs state durs has shape (n_mice, 3) for the three different types
# sort the array by the last column

sort_by = [0, 1, 2, 'sum']
fs = 34

for sb in sort_by:
    for normalize in [True, False]:
        abs_state_durs = abs_state_durs_true.copy()
        abs_state_durs_unnormalized = abs_state_durs_true.copy()
        # normalize each row
        if normalize:
            abs_state_durs = abs_state_durs / np.sum(abs_state_durs, axis=1)[:, np.newaxis]

        # sort
        if sb == 'sum':
            abs_state_durs_unnormalized = abs_state_durs_unnormalized[np.argsort(abs_state_durs.sum(1))]
            abs_state_durs = abs_state_durs[np.argsort(abs_state_durs.sum(1))]
        else:
            abs_state_durs_unnormalized = abs_state_durs_unnormalized[np.argsort(abs_state_durs[:, sb])]
            abs_state_durs = abs_state_durs[np.argsort(abs_state_durs[:, sb])]

        # plot each animal as a stacked bar plot, coloured by the state types in this order: green, blue, red
        fig = plt.figure(figsize=(15, 8))
        spec = gridspec.GridSpec(ncols=100, nrows=100, figure=fig)
        spec.update(hspace=0.)  # set the spacing between axes.
        ax0 = fig.add_subplot(spec[:20, 0:98])
        ax1 = fig.add_subplot(spec[25:, 0:98])
        for i, row in enumerate(abs_state_durs):
            ax1.bar(i, row[0], color='g')
            ax1.bar(i, row[1], color='b', bottom=row[0])
            ax1.bar(i, row[2], color='r', bottom=row[0] + row[1])
        ax0.bar(np.arange(len(abs_state_durs_unnormalized)), abs_state_durs_unnormalized.sum(1), color='grey')
        ax0.set_xlim(-1, len(abs_state_durs_unnormalized))
        ax0.set_xticks([])
        ax0.set_ylabel("# sessions", fontsize=fs-14)

        ax1.set_xlim(-1, len(abs_state_durs))
        ax1.set_xticks([])
        ax1.set_xlabel("Mice", fontsize=fs)
        # increase tick font size
        ax1.tick_params('both', labelsize=fs - 20)
        ax1.set_ylim(0, 1)

        # add text to the right of the bars on ax1
        ax1.text(len(abs_state_durs_unnormalized) + 1.5, 0.75, "Stage 3", color='red', fontsize=fs-5)
        ax1.text(len(abs_state_durs_unnormalized) + 1.5, 0.5, "Stage 2", color='blue', fontsize=fs-5)
        ax1.text(len(abs_state_durs_unnormalized) + 1.5, 0.25, "Stage 1", color='green', fontsize=fs-5)

        if normalize:
            ax1.set_ylabel("Session percentage", fontsize=fs)
        else:
            ax1.set_ylabel("Session number", fontsize=fs)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"state_durs_normalized={normalize}_sorted_by_{sb}.png")
        plt.show()