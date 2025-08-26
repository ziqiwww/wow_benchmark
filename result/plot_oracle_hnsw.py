import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True


def process_our_layered(data):
    result_entry = []
    # first sort by recall in descending order
    data = data[np.argsort(data[:, 1])[::-1]]
    for i in range(data.shape[0]):
        recall = data[i, 1]
        qps = data[i, 2]
        # if recall is less than some existing recall and qps is also less, skip
        good = True
        for j in range(len(result_entry)):
            entry = result_entry[j]
            cur_recall = entry[1]
            cur_qps = entry[2]
            if recall < cur_recall and qps < cur_qps:
                good = False
                break
            if recall == cur_recall and qps > cur_qps:
                result_entry[j] = data[i]
                good = False
                break
        if good:
            result_entry.append(data[i])
    return np.array(result_entry)


if __name__ == "__main__":
    """plot oracle hnsw and compare with OURS_PP"""
    from common_plot import (
        get_line_color,
        get_line_style,
        get_marker,
    )

    from common_plot import (
        get_x_y_data_with_metric,
        get_x_y_label_with_metric,
        get_db_name_beautify,
        get_method_beautify,
        get_z_order,
        get_y_ticks_ylabels,
    )

    from common_plot import (
        METHOD_LIST,
        OURS_PP,
        OURS_LAYERED,
        WST,
        IRANGE,
        SERF,
        HSIG,
        DIGRA,
        RECALL_THRESHOLD,
        RANGE_FRACTIONS,
        EVAL_METRIC,
    )
    EVAL_METRIC="dist-recall"

    fraction = 17
    DB_LIST = ["sift", "gist","arxiv", "wiki4m",  "deep10m"]
    METHODS_TO_PLOT = OURS_PP + IRANGE + HSIG + DIGRA + ["oracle_hnsw"]
    K = 10

    num_cols = 2

    labels = []
    handles = []

    fig, axes = plt.subplots(1, len(DB_LIST), figsize=(20, 2.6), dpi=150)
    axes = axes.flatten()
    x_label, y_label = get_x_y_label_with_metric(EVAL_METRIC, K)

    for dataset, ax, plot_i in zip(DB_LIST, axes, range(len(DB_LIST))):
        final_y_ticks, final_y_ticks_labels = [], []
        for method in METHODS_TO_PLOT:
            csv_file = os.path.join(
                f"{dataset}_log_csv_k{K}", method, f"{fraction}.csv"
            )
            if not os.path.exists(csv_file):
                continue
            print(f"loading {csv_file}")
            data = np.loadtxt(csv_file, delimiter=",")
            if method in OURS_PP or method in HSIG:
                data = data.reshape(-1, 7) 
            elif method in IRANGE or method in DIGRA:
                data = data.reshape(-1, 5)
            data = data[data[:, 1] >= RECALL_THRESHOLD]
            data = data[np.argsort(data[:, 1])]
            x, y = get_x_y_data_with_metric(data, EVAL_METRIC)
            y_ticks, y_ticks_labels = get_y_ticks_ylabels(y, True)
            if len(y_ticks) > len(final_y_ticks):
                final_y_ticks = y_ticks
                final_y_ticks_labels = y_ticks_labels
            (line,) = ax.plot(
                x,
                y,
                label=method,
                linestyle=get_line_style(method),
                marker=get_marker(method),
                markerfacecolor="none",
                markersize=8,
                color=get_line_color(method),
                zorder=get_z_order(method),
            )
            method = get_method_beautify(method)
            if method == "WoW (unordered)":
                method = "WoW"
            if method not in labels:
                handles.append(line)
                labels.append(method)
            
            # ax.set_title(f"{get_db_name_beautify(dataset)}", fontsize=20, )
        # ax.grid()

        x_label = f"Recall@{K}"
        ax.set_xlabel(x_label, fontsize=20, labelpad=0.1)
        if plot_i == 0:
            ax.set_ylabel(y_label, fontsize=20)

        # yscale should vary according to y_ticks
        ax.set_yscale("log", base=10)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # ax.set_yticks(final_y_ticks)
        # ax.set_yticklabels(final_y_ticks_labels)

        # ax.set_ylim([min(final_y_ticks), max(final_y_ticks)])

        ax.set_xlim([RECALL_THRESHOLD, 1.01])
        
        
        ax.text(
            0.05,
            0.85,
            "\\textbf{" + f"{get_db_name_beautify(dataset)}" + "}",
            fontsize=24,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=22,
        borderaxespad=0,
        borderpad=0,
        markerscale=1.2,
        ncol=len(METHODS_TO_PLOT),
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig("oracle_hnsw.pdf", dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"result saved to {os.path.abspath('oracle_hnsw.pdf')}")
    # plt.show()
