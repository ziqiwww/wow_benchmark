# """
# plot recall-qps figure, each figure is a query range, and figure contains all method
# """

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['text.usetex'] = True

if __name__ == "__main__":

    DB = "arxiv"
    # DB = "gist"
    DB = "gist"
    K = 10
    DB_DIR = f"{DB}_log_csv_k{K}"
    plot_legend = True
    use_log_scale = False

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
        RECALL_THRESHOLD,
        RANGE_FRACTIONS,
        EVAL_METRIC,
    )
    EVAL_METRIC="qps-recall"
    
    
    color_list = [
        "#cc9966",
        "#cccc66",
        "#99cc66",
        "#66cc99",
        "#6699cc",
        "#9966cc",
        "#cc6699",
        "#b83d3d",
        "#808080",
        "#3db8b8",
        "#cc9966"
    ]

    num_cols = 2  # Number of subplots per row
    num_rows = 2

    # Adjust figsize to make each subplot approximately square
    subplot_size = 5  # Size of each subplot (width and height)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 2.6),
        squeeze=False,
        dpi=150,
    )

    # Store handles and labels for the legend
    handles = []
    labels = []
    
    METHOD_LIST = [
        "spatt-pp-10-16-4-dy",
        "spatt-pp-10-16-4-st",
    ]
    RANGE_FRACTIONS = [1,4,7,10]
    for t_rng, ax, plot_i in zip(
        RANGE_FRACTIONS, axes.flatten(), range(len(RANGE_FRACTIONS))
    ):
        final_y_ticks, final_y_ticks_labels = [], []
        sel_ld = 0
        csv_file = os.path.join(DB_DIR, "spatt-pp-10-16-4-dy", f"{t_rng}.csv")
        if not os.path.exists(csv_file):
            # pass the current range if the method does not have the csv file
            continue
        print(f"loading {csv_file}")
        data = np.loadtxt(csv_file, delimiter=",")
        data = data.reshape((-1, 7))
        sel_ld = int(data[0, 6])
        print(f"rng:, {t_rng}, sel_ld: {sel_ld}")

        data = data[data[:, 1] >= RECALL_THRESHOLD]
        if data.shape[0] == 0:
            continue
        x, y = get_x_y_data_with_metric(data, EVAL_METRIC)
        y_ticks, y_ticks_labels = get_y_ticks_ylabels(y, True)
        if len(y_ticks) > len(final_y_ticks):
            final_y_ticks = y_ticks
            final_y_ticks_labels = y_ticks_labels
        # if EVAL_METRIC == "dist-recall":
        #     ax.set_yscale("log")
        sel_label_name = f"$l_d={sel_ld}$"
        (sel_line,) = ax.plot(
            x,
            y,
            label= sel_label_name,
            linestyle=get_line_style("spatt-pp-10-16-4-dy"),
            marker=get_marker("spatt-pp-10-16-4-dy"),
            markersize=8,
            markerfacecolor="none",
            # linewidth=2,
            color=color_list[sel_ld],
            zorder=sel_ld,
        )
        if (sel_ld, sel_label_name) not in labels:
            handles.append((sel_ld, sel_line))
            labels.append((sel_ld,sel_label_name))
        
        top_layer_list = list(range(0, 11))
        
        st_csv = os.path.join(DB_DIR, "spatt-pp-10-16-4-st", f"{t_rng}.csv")
        data = np.loadtxt(st_csv, delimiter=",")
        data = data.reshape((-1, 7))
        for top_layer in top_layer_list:
            if top_layer == sel_ld or top_layer <= 2:
                continue
            data_top_layer = data[data[:, 6] == top_layer]
            if data_top_layer.shape[0] == 0:
                continue
            data_top_layer = data_top_layer[data_top_layer[:, 1] >= RECALL_THRESHOLD]
            if data_top_layer.shape[0] == 0:
                continue
            data_top_layer = data_top_layer[np.argsort(data_top_layer[:, 1])]
            x, y = get_x_y_data_with_metric(data_top_layer, EVAL_METRIC)
            label_name = f"$l_d={top_layer}$"
            (line,) = ax.plot(
                x,
                y,
                label= label_name,
                linestyle=get_line_style("spatt-pp-10-16-4-dy"),
                marker=get_marker("spatt-pp-10-16-4-dy"),
                markersize=8,
                markerfacecolor="none",
                # linewidth=2,
                color=color_list[top_layer],
                zorder=top_layer,
            )
            if (top_layer, label_name) not in labels:
                handles.append((top_layer, line))
                labels.append((top_layer, label_name))
        x_label, y_label = get_x_y_label_with_metric(EVAL_METRIC, K)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if plot_i == 2 or plot_i == 3:
            ax.set_xlabel(x_label, fontsize=18, labelpad=0.2)
        if plot_i == 0 or plot_i == 2:
            ax.set_ylabel(y_label, fontsize=16)
        # add a small legend at the top right which only contains the sel_ld
        ax.legend(
            [sel_line],
            [f"Alg. 3: {sel_label_name}"],
            loc="upper right",
            fontsize=16,
            frameon=False,
            markerscale=0.7,
            labelspacing=0.2,
            handlelength=1.5,
            # place line to the right of the text
            borderaxespad=0.2,
            borderpad=0.2,
            handletextpad=0.2,
        )
        

        if use_log_scale:
            ax.set_yscale("log", base=10)
            ax.set_yticks(final_y_ticks)
            ax.set_yticklabels(final_y_ticks_labels)
            ax.set_ylim([min(final_y_ticks), max(final_y_ticks)])
        
        
        # set the right boundary of the figure to be more than 1.0
        ax.set_xlim([RECALL_THRESHOLD, 1.01])
        if t_rng == 17:
            ax.set_title(r"Mixed range fraction", fontsize=20, pad=0.2)
        else:
            ax.set_title(r"Range fraction: $2^{" + f"-{t_rng}" + "}$", fontsize=20, pad=0.2)
    # Hide any unused subplots
    for ax in axes.flatten()[len(RANGE_FRACTIONS) :]:
        ax.set_visible(False)
        
    # sort the legend by the order of the sel_ld
    handles = [x[1] for x in sorted(handles, key=lambda x: x[0])]
    labels = [x[1] for x in sorted(labels, key=lambda x: x[0])]

    # Add a legend to the figure and make it tight
    if plot_legend:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.07),
            fontsize=18,
            frameon=False,
            labelspacing=0.2,
            handlelength=1.5,
            borderaxespad=0,
            borderpad=0,
            handletextpad=0.2,
        )

    # set fig title with bold and larger font at the top using Times New Roman
    # fig.suptitle(f"{get_db_name_beautify(DB)}", fontsize=26, fontweight="bold", y=0.93)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.27, wspace=0.16)
    # 
    res_file = os.path.join(f"gist_layer_selection.pdf")
    plt.savefig(res_file, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"save figure to {res_file}")
    # plt.show()
