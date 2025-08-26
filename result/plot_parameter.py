import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        RECALL_THRESHOLD,
        RANGE_FRACTIONS,
        EVAL_METRIC,
    )

    fraction = 17
    DB = "deep10m" # gist, sift, arxiv, wiki4m, deep10m
    target_parameter = "efc"  # o, m, efc
    # use_qps_over_index_time = False
    K = 10
    
    def get_param_list(db, target_param):
        if target_param == "m":
            return [4, 8, 16, 24, 32, 48, 64, 128]
        if target_param == "efc":
            return [16, 32, 64, 128, 256, 512, 1024, 2048]
        if target_param != "o":
            raise ValueError(f"Unknown parameter {target_param} for db {db}") 
        if db == "gist":
            return [(19, 2), (10, 4), (8, 6), (7, 8), (6, 10), (6, 12), (5, 14), (5, 16)]
        if db == "sift":
            return [(19, 2), (10, 4), (8, 6), (7, 8), (6, 10), (6, 12), (5, 14), (5, 16)]
        if db == "arxiv":
            return [(21,2), (11,4), (8,6), (7,8), (7,10), (6,12), (6,14), (6,16)]
        if db == "wiki4m":
            return [(21,2), (11,4), (9,6), (7,8), (7,10), (6,12), (6,14), (6,16)]
        if db == "deep10m":
            return [(23,2), (12,4), (9,6), (8,8), (7,10), (7,12), (6,14), (6,16)]
        else:
            raise ValueError(f"Unknown db {db} for parameter {target_param}")
    
    def gen_param_list(db, target_parameter):
        default_wp_o = (10, 4)
        if db == "arxiv":
            default_wp_o = (11, 4)
        if db == "wiki4m":
            default_wp_o = (11, 4)
        if db == "deep10m":
            default_wp_o = (12, 4)
        if target_parameter == "o":
            return [
                (f"spatt-pp-{i_wp}-{16}-{i_o}-dy", i_o)
                for i_wp, i_o in get_param_list(db, target_parameter)
            ]
        elif target_parameter == "m":
            return [
                (f"spatt-pp-{default_wp_o[0]}-{i}-{default_wp_o[1]}-dy", i)
                for i in get_param_list(db, target_parameter)
            ]
        elif target_parameter == "efc":
            return [
                (f"spatt-pp-{default_wp_o[0]}-{16}-{default_wp_o[1]}-{i}-dy", i)
                for i in get_param_list(db, target_parameter)
            ]

    def get_building_time(t_param, t_value):
        index_csv = f"{DB}_varying_{t_param}.csv"
        data = np.loadtxt(index_csv, delimiter=",")
        for i in range(data.shape[0]):
            if data[i, 0] == t_value:
                return data[i, 1]
                return np.log2(data[i, 1])
        return 0

    def get_parameter_beautify(parameter):
        if parameter == "o":
            return "o"
        elif parameter == "m":
            return "m"
        elif parameter == "efc":
            return "\omega_c"
        else:
            raise ValueError(f"parameter {parameter} not found")

    num_cols = 2

    labels = []
    handles = []

    color_list = [
        "#cc6666",
        "#cccc66",
        "#66cc66",
        "#66cccc",
        "#6666cc",
        "#cc66cc",
        "#8a2e5c",
        "#2e8a5c",
        "#808080",
    ]

    width = 4
    hight = 2.5

    fig, axes = plt.subplots(2, 2, figsize=(width * 2, hight * 2), dpi=150)
    axes = axes.flatten()

    # axes [0-4] are bar plots

    for metric, use_qps_over_index_time, ax in zip(
        ["qps-recall", "qps-recall"], [False, True], axes
    ):
        final_y_ticks, final_y_ticks_labels = [], []
        for i, (method, para_value) in enumerate(
            gen_param_list(DB, target_parameter)
        ):
            csv_file = f"{DB}_log_csv_k{K}/{method}/{fraction}.csv"
            print(f"loading {csv_file}")
            data = np.loadtxt(csv_file, delimiter=",")
            rep_data = data.copy()
            data = data[data[:, 1] >= RECALL_THRESHOLD]
            data = data[np.argsort(data[:, 1])]
            o_name = f"${get_parameter_beautify(target_parameter)}={para_value}$"
            # if data.shape[0] != 0:
            x, y = get_x_y_data_with_metric(data, metric)
            if use_qps_over_index_time:
                # if qps-recall, divide y by building time
                building_time = get_building_time(target_parameter, para_value)
                y = y / building_time
            # y_ticks, y_ticks_labels = get_y_ticks_ylabels(y, True)

            # if len(y_ticks) > len(final_y_ticks):
            #     final_y_ticks = y_ticks
            #     final_y_ticks_labels = y_ticks_labels
            (line,) = ax.plot(
                x,
                y,
                label=o_name,
                color=color_list[i],
                zorder=i,
                linewidth=2.5,
                # linestyle=get_line_style(METHOD),
                # marker=get_marker(METHOD),
                # zorder=get_z_order(METHOD),
            )
            if o_name not in labels:
                handles.append(line)
                labels.append(o_name)
            # else:
            continue
                # if the method recall is les than RECALL_THRESHOLD, plot a small subsubplot at the left bottom of ax
            rep_data = rep_data[np.argsort(rep_data[:, 1])]
            rep_data = rep_data[rep_data[:, 1] >= 0.3]
            print(f"rep_data[-1, 1]: {rep_data[-1, 1]}")
            if rep_data[-1, 1] < RECALL_THRESHOLD:
                x, y = get_x_y_data_with_metric(rep_data, metric)
                if use_qps_over_index_time:
                    # if qps-recall, divide y by building time
                    building_time = get_building_time(target_parameter, para_value)
                    y = y / building_time
                y_ticks, y_ticks_labels = get_y_ticks_ylabels(y, True)
                print(f"plotting inset for {method}")
                if metric == "qps-recall":
                    cur_loc = "lower left"
                    cur_bbox_to_anchor = (0.01, 0.01, 1, 1)
                elif metric == "dist-recall":
                    cur_loc = "upper left"
                    cur_bbox_to_anchor = (0.01, -0.01, 1, 1)
                axin = inset_axes(
                    ax,
                    width="25%",
                    height="32%",
                    loc=cur_loc,
                    bbox_to_anchor=cur_bbox_to_anchor,
                    bbox_transform=ax.transAxes,
                )
                (line,) = axin.plot(
                    x,
                    y,
                    label=o_name,
                    color=color_list[i],
                    zorder=i,
                    linewidth=2.5,
                )
                
                axin.spines["top"].set_linewidth(0.1)
                axin.spines["right"].set_linewidth(0.1)

                # set axin limit to be [0.0, 0.7]
                # round to the upper bound of rep_data[-1, 1]
                axin.set_xlim([0.30, np.ceil(rep_data[-1, 1] * 10) / 10])
                # set xsticks to be [0.0, 0.35, 0.7]
                ticks = np.round(np.linspace(0.3, round(rep_data[-1, 1], 2), 3), 2)
                # add rep_data[-1, 1] to the ticks
                axin.set_xticks(ticks)
                axin.xaxis.tick_top()
                axin.yaxis.tick_right()
                if not use_qps_over_index_time:
                    axin.set_yticks(y_ticks)
                    axin.set_yticklabels(y_ticks_labels)
                    axin.set_yscale("log", base=10)
                axin.tick_params(axis="both", which="major", labelsize=7, pad=0, length=1.5)
                axin.tick_params(axis="both", which="minor", labelsize=7, pad=0, length=1)
                if o_name not in labels:
                    handles.append(line)
                    labels.append(o_name)

        x_label, y_label = get_x_y_label_with_metric(metric, K)
        # ax.set_xlabel(x_label, fontsize=16)
        # ax.set_ylabel(y_label, fontsize=12)
        if metric == "qps-recall":
            if use_qps_over_index_time:
                title = "(ii) QPS/IT-Recall@10"
            else:
                title = "(i) QPS-Recall@10"
        else:
            title = "DC-Recall@10"
        ax.set_title(title, fontsize=16)
        # ax.grid()
        # if not use_qps_over_index_time: 
        #     ax.set_yscale("log", base=10)
        #     ax.set_yticks(final_y_ticks)
        #     ax.set_yticklabels(final_y_ticks_labels)
        #     ax.set_ylim([min(final_y_ticks), max(final_y_ticks)])
            
        ax.tick_params(axis="both", which="major", labelsize=12)
        
        # keep 2 decimal places for xticks
        ax.set_xticks(np.round(np.linspace(0.9, 1, 3), 2))

        ax.set_xlim([RECALL_THRESHOLD, 1.01])

    # first plot the building information
    data = np.loadtxt(f"{DB}_varying_{target_parameter}.csv", delimiter=",")
    for i, (indicator, ax) in enumerate(
        zip(
            [
                "(iii) Indexing Time (s)",
                "(iv) Index Size (MB)",
                # "(v) DC per Insertion",
                # "(vi) Average Outdegree",
            ],
            axes[2:],
        )
    ):
        x = data[:, 0]
        a = np.arange(0, len(x))
        y = data[:, i + 1]
        # plot bars with different colors
        # bar style: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
        ax.bar(
            a,
            y,
            alpha=0.8,
            # fill=False,
            edgecolor="black",
            # hatch="//",
            color=color_list,
            linewidth=1.5,
            zorder=2,
        )

        # ax.set_xlabel("Boosting base $o$", fontsize=16)
        ax.set_title(indicator, fontsize=16)
        ax.set_xticks(a)
        ax.set_xticklabels([f"{int(i)}" for i in x])
        ax.tick_params(axis="both", which="major", labelsize=12)

    # 2 lines of legend
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.11),
        fontsize=20,
        frameon=False,
        ncol=len(gen_param_list(DB,target_parameter)) // 2,
        labelspacing=0.2,
        handlelength=1,
        borderaxespad=0,
        borderpad=0,
        handletextpad=0.2,
        
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.14)
    plt.savefig(f"{DB}_varying_{target_parameter}.pdf", dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"result saved to {os.path.abspath(f'{DB}_varying_{target_parameter}.pdf')}")
