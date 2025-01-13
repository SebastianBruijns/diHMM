import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
subject = 'KS014'
df = pd.read_csv("./summarised_sessions/{}{}/{}_{}_fit_info.csv".format(str(0.25).replace('.', '_'), '', subject, 'prebias'))


# Example data: Replace these arrays with your actual data
training_days = np.arange(1, 1 + len(np.unique(df.session)))  # Example training days
contrast = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]  # Contrast values


rightward_choices = np.ones((len(contrast), len(training_days)))
rightward_choices *= np.nan

for i, day in enumerate(training_days):
    conts = df[df.session == i].left_contrast - df[df.session == i].right_contrast
    choices = df[df.session == i].choice
    for cont in np.unique(conts):
        rightward_choices[contrast_to_num[cont], i] = np.mean(1 - choices[conts == cont]) * 100


# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Create a heatmap with the data
heatmap = sns.heatmap(
    rightward_choices,
    xticklabels=training_days,  # Labels for training days
    yticklabels=contrast,  # Labels for contrast
    cmap='coolwarm',  # Colormap for rightward choices
    cbar_kws={'label': 'Rightward choices (%)'},  # Label for colorbar
    ax=ax,
    vmin=0, vmax=100
)

# Increase colorbar label font size
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.label.set_size(18)  # Set font size of colorbar label

# Set axis labels
ax.set_xlabel("Training day", fontsize=20)
ax.set_ylabel("Contrast", fontsize=20)

# Show the plot
plt.tight_layout()
plt.savefig("ks014_heatmap.png")
plt.show()