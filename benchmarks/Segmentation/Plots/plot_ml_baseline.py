import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict

# plt.rcParams["font.family"] = "Times New Roman"


def plot_baseline(cat_to_color, name2_acc, name2_maP, Baseline_cat, Y_lim, Y_lim2, distance=1.5):

    distance = 1.5  # Distance between each bar
    x = np.arange(0, len(name2_maP), 1).astype("float")*distance
    x_idx = 0

    ticks = []
    ImageNet_acc = []
    # Plot bars
    fig, ax1 = plt.subplots()
    for category, methods in Baseline_cat.items():
        for i, method in enumerate(methods):
            if i == 0:

                ax1.bar(x[x_idx], name2_maP[method],
                        color=cat_to_color[category], label=category)
            else:
                ax1.bar(x[x_idx], name2_maP[method],
                        color=cat_to_color[category])
            # Add the text of value
            ax1.text(x[x_idx], name2_maP[method], " "+str(name2_maP[method]),
                     color='k', ha="center", va='bottom', fontweight='bold')

            ticks.append(method)
            ImageNet_acc.append(name2_acc[method])
            x_idx += 1
    # Plots for linear classification
    # ax2 = ax1.twinx()
    # ax2.plot(x[:], ImageNet_acc, color='k', marker="o")
    # for i, j in zip(x[:], ImageNet_acc):
    #     ax2.annotate(str(j), xy=(i, j+0.3), ha="center",
    #                  va='bottom', fontweight='bold', color='k')
    plt.xticks(x, ticks)
    plt.xlim(-1, len(ticks)*distance)
    ax1.set_ylim(Y_lim[0], Y_lim[1])
    # ax2.set_ylim(Y_lim2[0], Y_lim2[1])
    ax1.legend(loc=1)
    plt.show()


##################
# VOC
##################
# The category for each baeline method
Baseline_cat = OrderedDict(
    {
        "Clustering": ["SwAV"],
        "Distillation": ["BYOL"],
        "Contrastive": ["MoCov2"],
        "Region-level": ["DenseCL", "DetCo"]
    }
)
name2_maP = {
    "SwAV": 88.9,
    "BYOL": 88.8,
    # "SCRL": 87.3,
    # "MoCov2": 87.2,
    "DenseCL": 83.9,
    "DetCo": 86.3,
    "MoCov2": 85.6,
}
name2_acc = {
    "SwAV": 75.3,
    "BYOL": 74.3,
    # "SCRL": 70.7,
    # "MoCov2": 71.7,
    "DenseCL": 63.8,
    "DetCo": 68.5,
    "MoCov2": 67.5,
}
cat_to_color = {
    "Contrastive": "skyblue",
    "Clustering": "khaki",
    "Distillation": 'lightcoral',
    "Region-level": "palegreen"
}
plot_baseline(cat_to_color, name2_acc, name2_maP,
              Baseline_cat, (80, 90), (60, 80))


##################
# COCO
##################
# The category for each baeline method
Baseline_cat = OrderedDict(
    {
        "Clustering": ["SwAV"],
        "Distillation": ["BYOL"],
        "Contrastive": ["MoCov2", "MoCov2\n-200ep"],
        "Region-level": ["SCRL", "DenseCL", "DetCo"]
    }
)
name2_maP = {
    "SwAV": 69.3,
    "BYOL": 67.2,
    "SCRL": 64.3,
    "MoCov2": 65.3,
    "DenseCL": 61.4,
    "DetCo": 65.0,
    "MoCov2\n-200ep": 0.,
}
name2_acc = {
    "SwAV": 75.3,
    "BYOL": 74.3,
    "SCRL": 70.7,
    "MoCov2": 71.7,
    "DenseCL": 63.8,
    "DetCo": 68.5,
    "MoCov2\n-200ep": 67.5,
}
cat_to_color = {
    "Contrastive": "skyblue",
    "Clustering": "gold",
    "Distillation": 'crimson',
    "Region-level": "palegreen"
}
plot_baseline(cat_to_color, name2_acc, name2_maP,
              Baseline_cat, (55, 75), (60, 80))

######
# Pixel Level vs non Pixel level
######
Baseline_cat = OrderedDict(
    {
        "Distillation": ["BYOL"],
        "Region-level": ["SCRL"]
    }
)
name2_maP = {
    "BYOL": 88.8,
    "SCRL": 87.3,
}
name2_acc = {
    "BYOL": 74.3,
    "SCRL": 70.7,
}
cat_to_color = {
    "Contrastive": "skyblue",
    "Clustering": "gold",
    "Distillation": 'crimson',
    "Region-level": "palegreen"
}
plot_baseline(cat_to_color, name2_acc, name2_maP,
              Baseline_cat, (80, 90), (60, 80))
Baseline_cat = OrderedDict(
    {
        "Contrastive": ["MoCov2\n-200ep"],
        "Region-level": ["DenseCL", "DetCo"]
    }
)
name2_maP = {
    "DenseCL": 83.9,
    "DetCo": 86.3,
    "MoCov2\n-200ep": 85.6,
}
name2_acc = {
    "DenseCL": 63.8,
    "DetCo": 68.5,
    "MoCov2\n-200ep": 67.5,
}
cat_to_color = {
    "Contrastive": "skyblue",
    "Clustering": "gold",
    "Distillation": 'crimson',
    "Region-level": "palegreen"
}
plot_baseline(cat_to_color, name2_acc, name2_maP,
              Baseline_cat, (80, 90), (60, 80))
