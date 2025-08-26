import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['text.usetex'] = True

# plot 5 figures , each is the histogram of the hop layer, each bar has two parts, the lower part is the low layer, the upper part is the high layer

WO_LIST = [
    "wow",
    # "wow_wo_cn",
    "wow_wo_next",
    # "wow_wo_cn_next",
    # "wow_wo_sel",
]

def get_ablation_name_beautify(name):
        if name == "wow":
            return "WoW"
        if name == "wow_wo_cn":
            return "w/o $c_n$"
        if name == "wow_wo_next":
            return "w/o early-stop"
        if name == "wow_wo_cn_next":
            return "w/o ($c_n$ + next)"
        if name == "wow_wo_sel":
            return "w/o $l_d$"
        return name

# read from "gist_hop_layer_{WO}.txt", each txt is formatted: hop,high_layer,low_layer
def read_hop_layer(WO):
    hop_layer = []
    with open(f"gist_hop_layer_{WO}.txt", "r") as f:
        print(f"loading gist_hop_layer_{WO}.txt")
        for line in f:
            hop, high_layer, low_layer = map(int, line.strip().split(","))
            # if low_layer < 0, set to 0
            low_layer = max(0, low_layer)
            assert low_layer >= 0
            hop_layer.append((hop, high_layer, low_layer))
    return hop_layer

figure = plt.subplots(len(WO_LIST), 1, figsize=(4*len(WO_LIST), 4), constrained_layout=True)

sub_title_id = ["(a) ", "(b) ", "(c) ", "(d) ", "(e) "]

for i, WO in enumerate(WO_LIST):
    hop_layer = read_hop_layer(WO)
    hop_layer = np.array(hop_layer)
    ax = figure[1][i]
    assert hop_layer.shape[1] == 3
    # assert low and high are non-negative
    assert np.all(hop_layer[:, 1] >= 0)
    assert np.all(hop_layer[:, 2] >= 0)
    ax.bar(hop_layer[:400, 0], hop_layer[:400, 1], label="high_layer", color="#8ec1da", align='edge', width=1.0, linewidth=0) #"#c46666" blue: "#082a54"
    ax.bar(hop_layer[:400, 0], hop_layer[:400, 2], label="low_layer", color="#cde1ec", align='edge', width=1.0, linewidth=0) # "#f2c45f" blue: "#cde1ec"
    
    # plot average (high_layer + low_layer) / 2
    ax.plot(hop_layer[:400, 0], (hop_layer[:400, 1] + hop_layer[:400, 2]) / 2, label="average", color="#cc6666", linestyle="-", linewidth=0.5)
    print(hop_layer[:400, 1], hop_layer[:400, 2])
    # ax.set_xlabel("Hop")
    # title = r"\textbf{" + sub_title_id[i] + get_ablation_name_beautify(WO) + r"}"
    # ax.set_title(title, fontsize=14)
    ax.text(
            0.01,
            0.15,
            "\\textbf{" + f"{get_ablation_name_beautify(WO)}" + "}",
            fontsize=24,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
    # ylimit to 10, and display the maximum x value on the x axis
    ax.set_ylim([0, 10])
    ax.set_xlim([0, 400])
    # ax.set_xlim([0, max(hop_layer[:, 0])])
    ax.tick_params(axis="both", which="major", labelsize=12, pad=1)
    
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel("Layer range", fontsize=16, labelpad=0)
    if i == len(WO_LIST) - 1:
        ax.set_xlabel("Hop number $i$ during graph traversal", fontsize=18, labelpad=1)
    # xticks should include 200, 400, 600, ... , max(hop_layer[:, 0]) + 1
    xticks = [i for i in range(0, 401, 100)]
    # xticks.append(max(hop_layer[:, 0]))
    ax.set_xticks(xticks)
    # ax.set_xticks(range(0, max(hop_layer[:, 0]) + 1, 200))
    # ax.legend()
    
    
    
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.savefig("hop_layer.pdf", dpi=150, bbox_inches="tight", pad_inches=0)
print(f"result saved to {os.path.abspath('hop_layer.pdf')}")
