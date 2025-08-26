# """
# plot recall-qps figure, each figure is a query range, and figure contains all method
# """

# import matplotlib.pyplot as plt

# import numpy as np
# import os

# import glob
# import re

# if __name__ == "__main__":
#     METHOD_LIST = [
#         "prefiltering",
#         "postfiltering",
#         "irangegraph",
#         "vamana-tree",
#         "optimized-postfiltering",
#         "smart-combined",
#         "three-split",
#         "super-postfiltering",
#     ]
#     RANGE_FRACTIONS = range(0, 10)
#     # plot sub figures in 1 * len(RANGE_FRACTIONS) grid
#     for t_rng in RANGE_FRACTIONS:
#         for i, method in enumerate(METHOD_LIST):
#             csv_file = os.path.join(method, f"{t_rng}.csv")
#             if not os.path.exists(csv_file):
#                 raise ValueError(f"{csv_file} does not exist")
#             print(f"loading {csv_file}")
#             data = np.loadtxt(csv_file, delimiter=",")
#             # reshape
#             if method == "irangegraph":
#                 data = data.reshape((-1, 5))
#             else:
#                 data = data.reshape((-1, 3))
#             # plot all method into one figure with different color line
#             # set recall range: 0.9 - 1.0
#             data = data[data[:, 1] >= 0.9]
#             # sort by recall
#             data = data[np.argsort(data[:, 1])]
#             if i == 0:
#                 plt.figure()
#             if method == "prefiltering":
#                 plt.plot(data[:, 1], data[:, 2], label=method, linestyle="--", color="black")
#             else:
#                 plt.plot(data[:, 1], data[:, 2], label=method)
#         plt.legend()
#         plt.xlabel("Recall")
#         plt.ylabel("Queries Per Second")
#         plt.title(f"Query Range {t_rng}")
#         plt.savefig(f"{t_rng}.png")
#         plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['text.usetex'] = True


if __name__ == "__main__":

    # DB = "arxiv"
    # DB = "gist"
    # DB = "sift"
    # DB = "sift_open"
    # DB="deep10m"
    # DB = "wiki4m"
    K = 10

    ax_height = 2.6

    use_log_scale = False
    
    DB_LIST = ["sift", "gist"]
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
        ACORN,
        MILVUS,
        DIGRA,
        DSG,
        RECALL_THRESHOLD,
        RANGE_FRACTIONS,
        EVAL_METRIC,
    )
    EVAL_METRIC = "qps-recall"  # Set the evaluation metric to qps-recall
    
    def method_filter(method):
        # Filter methods based on the evaluation metric
        if method in OURS_PP or method in HSIG or method in ACORN or method == "prefiltering" or method == "postfiltering" or method in DIGRA:
            return True
        return False

    num_cols = 5  # Number of subplots per row
    num_rows = len(DB_LIST)  # Calculate number of rows needed

    # Adjust figsize to make each subplot approximately square
    subplot_size = 5  # Size of each subplot (width and height)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * ax_height),
        squeeze=False,
        dpi=150,
    )

    # Store handles and labels for the legend
    handles = []
    labels = []
    for i_row, DB in enumerate(DB_LIST):
        plot_x_label = DB == "gist"
        plot_ax_title = DB == "sift"
        for i_col, cur_c in enumerate([100, 500, 1000, 5000, 10000]):
            DB_DIR = f"{DB}_log_csv_k{K}_common"
            ax = axes.flatten()[i_row * num_cols + i_col]
            final_y_ticks, final_y_ticks_labels = [], []
            for i, method in enumerate(METHOD_LIST):
                if not method_filter(method):
                    continue
                csv_file = os.path.join(DB_DIR, method, f"c{cur_c}.csv")
                if not os.path.exists(csv_file):
                    # pass the current range if the method does not have the csv file
                    continue
                print(f"loading {csv_file}")
                data = np.loadtxt(csv_file, delimiter=",")
                if method == "prefiltering":
                    # print a yellow star for prefiltering, whose data[0] is 1 and data[1] is qps
                    star = ax.scatter(
                        1,
                        data[1],
                        label=method,
                        color=get_line_color(method),
                        marker="*",
                        s=150,
                        zorder=get_z_order(method),
                    )
                    y_ticks, y_ticks_labels = get_y_ticks_ylabels(data[1], False)
                    if get_method_beautify(method) not in labels:
                        handles.append(star)
                        labels.append(get_method_beautify(method))
                    continue
                # reshape
                if (
                    
                    method == "oracle_hnsw"
                    or method in HSIG
                    or method == "postfiltering"
                ):
                    data = data.reshape((-1, 7))

                elif method in IRANGE or method in SERF or method in WST or method in DIGRA or method in DSG or method in OURS_LAYERED or method in OURS_PP:
                    data = data.reshape((-1, 5))
                elif method in ACORN or method in MILVUS:
                    data = data.reshape((-1, 3))
                else:
                    raise ValueError(f"method {method} not found")
                # plot all method into one figure with different color line
                # set recall range: x - 1.0
                # deep copy the data
                # rep_data = data.copy()
                data = data[data[:, 1] >= RECALL_THRESHOLD]
                data = data[np.argsort(data[:, 1])]
                if data.shape[0] == 0:
                    continue
                x, y = get_x_y_data_with_metric(data, EVAL_METRIC)
                y_ticks, y_ticks_labels = get_y_ticks_ylabels(y, False)
                if len(y_ticks) > len(final_y_ticks):
                    final_y_ticks = y_ticks
                    final_y_ticks_labels = y_ticks_labels
                # if EVAL_METRIC == "dist-recall":
                #     ax.set_yscale("log")
                (line,) = ax.plot(
                    x,
                    y,
                    label=method,
                    linestyle=get_line_style(method),
                    marker=get_marker(method),
                    markersize=8,
                    markerfacecolor="none",
                    # linewidth=2,
                    color=get_line_color(method),
                    zorder=get_z_order(method),
                )
                label = get_method_beautify(method)
                if label not in labels:
                    handles.append(line)
                    labels.append(label)

            x_label, y_label = get_x_y_label_with_metric(EVAL_METRIC, K)
            ax.tick_params(axis="both", which="major", labelsize=14)
            if plot_x_label:
                ax.set_xlabel(x_label, fontsize=20, labelpad=0.5)
            if i_col == 0:
                ax.set_ylabel(y_label, fontsize=20)

            if use_log_scale:
                ax.set_yscale("log", base=10)
                # ax.set_yticks(final_y_ticks)
                # ax.set_yticklabels(final_y_ticks_labels)

                ax.set_ylim([min(final_y_ticks), max(final_y_ticks)])

            # set the right boundary of the figure to be more than 1.0
            ax.set_xlim([RECALL_THRESHOLD, 1.01])
            # set lighter color for the grid
            # ax.grid(linewidth=0.5)
            # if query range is 17, it is a mixed work load from 0 to 2 ^ -10, else it is a single query range, use latex to render the query range
            if plot_ax_title:
                ax.set_title(f"\# of unique values: {cur_c}", fontsize=20, pad=0.5)
                

            # instead of plotting titles at the top of images, we plot the title at the top right in the first subplot
            if i_col == 0:
                ax.text(
                    0.96,
                    0.85,
                    "\\textbf{" + f"{get_db_name_beautify(DB)}" + "}",
                    fontsize=24,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                )

    # Add a single legend for the whole figure
    # rename the legend to be more readable

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.04),
        fontsize=22,
        markerscale=1.2,
        frameon=False,
        labelspacing=0,
        borderaxespad=0,
        borderpad=0,
        handletextpad=0.2,
        columnspacing=1.4,
        
    )

    # set fig title with bold and larger font at the top using Times New Roman
    # fig.suptitle(f"{get_db_name_beautify(DB)}", fontsize=26, fontweight="bold", y=0.93)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.14,wspace=0.18)
    res_file = f"qps-recall-duplicate.pdf"
    plt.savefig(res_file, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"save figure to {res_file}")
    # plt.show()
