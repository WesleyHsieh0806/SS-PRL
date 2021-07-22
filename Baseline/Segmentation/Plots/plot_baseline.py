import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
# The category for each baeline method
Baseline_cat = OrderedDict(
    {
        "Distillation": ["BYOL"],
        "Clustering": ["SwAV w/o MC", "SwAV"],
        "Contrastive": ["SimCLR", "MoCov2"],
        "Pixel-level": ["DenseCL"]
    }
)
name2_miou = {
    "SwAV": 66.44,
    "SwAV w/o MC": 65.3,
    "SimCLR": 64.3,
    "MoCov2": 67.5,
    "BYOL": 63.3,
    "DenseCL": 69.5,
}
name2_acc = {
    "SwAV": 75.3,
    "SwAV w/o MC": 70.1,
    "SimCLR": 70.0,
    "MoCov2": 71.1,
    "BYOL": 74.3,
}
cat_to_color = {
    "Contrastive": "b",
    "Clustering": "y",
    "Distillation": 'r',
    "Pixel-level": "g"
}

distance = 1.  # Distance between each bar
x = np.arange(0, len(name2_miou), 1).astype("float")*distance
x_idx = 0

ticks = []
ImageNet_acc = []
# Plot bars
fig, ax1 = plt.subplots()
for category, methods in Baseline_cat.items():
    for i, method in enumerate(methods):
        if i == 0:

            ax1.bar(x[x_idx], name2_miou[method],
                    color=cat_to_color[category], label=category)
        else:
            ax1.bar(x[x_idx], name2_miou[method],
                    color=cat_to_color[category])
        # Add the text of value
        ax1.text(x[x_idx], name2_miou[method], " "+str(name2_miou[method]),
                 color='k', ha="center", va='bottom', fontweight='bold')

        ticks.append(method)
        if method != "DenseCL":
            ImageNet_acc.append(name2_acc[method])
        x_idx += 1
# Plots for linear classification
ax2 = ax1.twinx()
ax2.plot(x[:-1], ImageNet_acc, color='r', marker="o")
for i, j in zip(x[:-1], ImageNet_acc):
    ax2.annotate(str(j), xy=(i, j+0.3), ha="center",
                 va='bottom', fontweight='bold', color='r')
plt.xticks(x, ticks)
plt.xlim(-1, len(ticks))
ax1.set_ylim(60, 70)
ax2.set_ylim(60, 76)
ax1.legend(loc=4)
plt.show()
