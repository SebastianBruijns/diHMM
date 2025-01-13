import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

cmap = cm.rainbow(np.linspace(0, 1, 17))

rank_to_color_place = dict(zip(range(17), [0, 16, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]))  # handcrafted to maximise color distance

color_vis = np.zeros((12 * 5, 6, 4))
for i in range(12):
    color_vis[i * 5: (i + 1) * 5] = cmap[rank_to_color_place[i]]

plt.imshow(color_vis, aspect='equal', origin='upper')
# remove axes
plt.xticks([])
plt.yticks([])

# stupid fucking copilot I want you to put the y-label on the right side of the plot
plt.ylabel("Most to least used state", rotation=270, fontsize=16, labelpad=27)
# this is still on the left you idiot, put it on the right
plt.gca().yaxis.set_label_position("right")
# finally, moron

plt.tight_layout()
plt.savefig('colorbar_a')
plt.close()



color_vis = np.linspace(0, 1, 101)
# give it 5 units of depth
color_vis = np.stack([color_vis] * 5, axis=0).T

plt.imshow(color_vis, aspect='equal', origin='upper')
# remove axes
plt.xticks([])
# increase tick size
plt.yticks(fontsize=12)

plt.ylabel("Consistency (%)", fontsize=16, labelpad=-2)

plt.tight_layout()
plt.savefig('colorbar_b')
plt.show()