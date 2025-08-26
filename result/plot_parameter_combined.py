import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter,FuncFormatter,FixedLocator

# --- Matplotlib Preamble ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'


def process_our_layered(data):
    """
    This function appears unused in the main script but is kept as requested.
    It filters data points to keep only the pareto-optimal ones (recall vs. qps).
    """
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
    # Assuming common_plot.py exists and provides this function
    # If not, you can create a placeholder like:
    # def get_db_name_beautify(db):
    #     return {"sift": "SIFT1M", "gist": "GIST1M", "arxiv": "Arxiv", "wiki4m": "Wikipedia4M", "deep10m": "DEEP10M"}.get(db, db)
    from common_plot import get_db_name_beautify

    # --- Plotting Configuration ---
    fraction = 17
    K = 10
    
    # --- Helper Functions for Parameter Configurations ---
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
        if db == "arxiv" or db == "wiki4m":
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
        
    # --- Helper Functions for Data Extraction ---
    def get_building_time(db, t_param, t_value):
        index_csv = f"{db}_varying_{t_param}.csv"
        data = np.loadtxt(index_csv, delimiter=",")
        # Column 0: param value, Column 1: build time
        row = data[data[:, 0] == t_value]
        return row[0, 1] if row.size > 0 else 0
    
    def get_index_size(db, t_param, t_value):
        index_csv = f"{db}_varying_{t_param}.csv"
        data = np.loadtxt(index_csv, delimiter=",")
        # Column 0: param value, Column 2: index size in MB
        row = data[data[:, 0] == t_value]
        return row[0, 2] if row.size > 0 else 0

    def get_parameter_beautify(parameter):
        if parameter == "o":
            return r"$o$"
        elif parameter == "m":
            return r"$m$"
        elif parameter == "efc":
            return r"$\omega_c$"
        else:
            raise ValueError(f"parameter {parameter} not found")
        
    # --- Plotting Setup ---
    labels = []
    handles = []

    color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    marker_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    width = 3
    hight = 2

    fig, axes = plt.subplots(5, 3, figsize=(width * 3, hight * 5), dpi=150)
    axes = axes.flatten()

    db_list = ["sift", "gist", "arxiv", "wiki4m", "deep10m"]
    db_color_map = {db: color for db, color in zip(db_list, color_list)}
    db_marker_map = {db: marker for db, marker in zip(db_list, marker_list)}

    # --- Main Plotting Loop ---
    # Rows: varying parameter (efc, m, o)
    # Columns: metric (QPS, Recall, Indexing Time, Index Size)
    for i_row, metric in enumerate(["QPS", "DC", "Recall@K", "Indexing time (s)", "Index size (GB)"]):
        for i_col, target_param in enumerate(["efc", "m", "o"]):
            ax = axes[i_row * 3 + i_col]

            for db in db_list:
                para_list = gen_param_list(db, target_param)
                x_data, y_data = [], []

                for csv_dir, p_value in para_list:
                    y_val = None
                    try:
                        if metric in ["QPS", "DC", "Recall@K"]:
                            csv_file = f"{db}_log_csv_k{K}/{csv_dir}/{fraction}.csv"
                            data = np.loadtxt(csv_file, delimiter=",", ndmin=2)
                            print(f"loading csv {csv_file}")
                            # Assumed CSV format: [num_threads, recall, qps, ...]
                            if metric == "Recall@K":
                                y_val = data[data[:, 0] == 1000, 1]
                            elif metric == "QPS":
                                y_val = data[data[:, 0] == 1000, 2]
                            elif metric == "DC":
                                y_val = data[data[:, 0] == 1000, 3]
                        
                        elif metric == "Indexing time (s)":
                            y_val = get_building_time(db, target_param, p_value)

                        elif metric == "Index size (GB)":
                            # Convert size from MB to GB
                            y_val = get_index_size(db, target_param, p_value)
                    
                    except (FileNotFoundError, IndexError, OSError) as e:
                        print(f"Warning: Could not load data for db={db}, p_val={p_value}, metric={metric}. Error: {e}")
                        continue
                    
                    if y_val is not None and y_val > 0:
                        x_data.append(p_value)
                        y_data.append(y_val)
                
                if not x_data:
                    continue

                # Sort data by x-values for a clean line plot
                sorted_indices = np.argsort(x_data)
                x_data_sorted = np.array(x_data)[sorted_indices]
                y_data_sorted = np.array(y_data)[sorted_indices]
                if metric == "Index size (GB)":
                    y_data_sorted = y_data_sorted / 1024.0
                if metric == "DC":
                    y_data_sorted = y_data_sorted / 1000.0
                # --- This is CORRECT ---
                if metric == "QPS":
                    y_data_sorted = y_data_sorted / 100.0 # You plot the value `6`

                line, = ax.plot(
                    x_data_sorted,
                    y_data_sorted,
                    label=get_db_name_beautify(db),
                    color=db_color_map[db],
                    marker=db_marker_map[db],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=8,
                    markerfacecolor="none"
                )

                # Collect handles for the legend only from the first subplot
                if i_row == 0 and i_col == 0:
                    handles.append(line)
                    labels.append(get_db_name_beautify(db))
            
            ax.tick_params(axis='y', which='major', labelsize=9, pad=0, length=2)
            ax.tick_params(axis='y', which='minor', labelsize=9, pad=0, length=2)
            ax.tick_params(axis='x', which='minor', labelsize=14, pad=0, length=2)
            ax.tick_params(axis='x', which='major', labelsize=14, pad=0, length=2)
            integer_major_formatter = FuncFormatter(lambda x, pos: f'{int(x)}')
# Formatter for minor ticks: show ONLY 3 and 6 as integers
            integer_minor_formatter = FuncFormatter(lambda x, pos: f'{int(x)}' if round(x) in [3, 6] else '')
            # --- Axis Formatting for each Subplot ---
            ax.xaxis.set_major_formatter(integer_major_formatter)
            
            if target_param == "m":
                # Step 1: Define the locations for the minor ticks.
                # minor_tick_locations = [16, 32, 48]
                
                # # Step 2: Use FixedLocator to place minor ticks at these exact positions.
                # ax.xaxis.set_minor_locator(FixedLocator(minor_tick_locations))
                
                # # Step 3: Use a formatter to ensure these minor ticks get labels.
                # ax.xaxis.set_minor_formatter(integer_major_formatter)
                ax.set_xticks([4, 16, 32, 48, 64, 128])
            
            if target_param == 'o':
                ax.set_xticks([2,4,6,8,10,12,14,16])

            if metric != "Recall@K":
                # ax.ticks
                ax.yaxis.set_major_formatter(integer_major_formatter) # Use integer formatter
                
                ax.yaxis.set_minor_formatter(integer_minor_formatter) # Use integer formatter
                
                ax.tick_params(axis='y', which='major', labelsize=13, pad=0, length=2)
                ax.tick_params(axis='y', which='minor', labelsize=13, pad=0, length=2)
            else:
                one_decimal_formatter = FuncFormatter(lambda x, pos: f'{x:.1f}')
                two_decimal_formatter= FuncFormatter(lambda x, pos: f'{x:.2f}')
                if target_param == "o":
                    ax.set_ylim(bottom=0.98, top=1.001) # Give a little padding
                    # Set the exact tick locations
                    ax.set_yticks([0.98, 0.99, 1.0])
                    # ax.yaxis.set_minor_locator(FixedLocator([0.97, 0.98, 0.99, 1.0]))
                    ax.yaxis.set_major_formatter(two_decimal_formatter)
                else:
                    ax.set_ylim(bottom=0.89, top=1.01) # Give a little padding
                    # Set the exact tick locations
                    ax.set_yticks([0.9, 1.0])
                    ax.yaxis.set_minor_locator(FixedLocator([0.95]))
                    
                    # Apply the custom formatter to show one decimal place
                    ax.yaxis.set_major_formatter(two_decimal_formatter)
                    
                    ax.yaxis.set_minor_formatter(two_decimal_formatter)
                ax.tick_params(axis='y', which='both', labelrotation=45)
                    
            if metric in ["QPS", "Indexing time (s)"]:
                ax.set_yscale("log")
                if metric == "QPS":
                    # ax.yaxis.set_major_formatter(integer_major_formatter) 
                    # # Use integer formatter
                    ax.tick_params(axis='y', which='major', labelleft=False)
                    ax.yaxis.set_minor_formatter(integer_minor_formatter) # Use integer formatter
                # # Prevent formatter from using offsets (e.g., "+1x10^4")
                # ax.yaxis.get_major_formatter().set_useOffset(False)

            if i_row == 5-1:
                # Set titles for the top row of plots
                x_label = f"{get_parameter_beautify(target_param)}"
                ax.set_xlabel(x_label, fontsize=20, labelpad=0)
            
            if i_col == 0:
                # Set Y-axis labels for the first column
                ylabel_text = metric.replace("@K", f"@{K}")
                cus_labpad = 0
                if metric == "QPS":
                    ylabel_text = r"QPS ($\times 10^2$)"
                    cus_labpad = 4
                if metric == "DC":
                    ylabel_text = r"DC ($\times 10^3$)"
                    cus_labpad = 4
                ax.set_ylabel(ylabel_text, fontsize=17, labelpad=cus_labpad)
            else:
                # ax.set_yticklabels([])
                pass

            # if i_row == 2:
            #     # Set X-axis labels for the bottom row
            #     ax.set_xlabel("Parameter Value", fontsize=16)
            # else:
            #     ax.set_xticklabels([])
                
            # ax.grid(True, which="both", ls="--", c='0.85')

    # --- Final Figure Adjustments ---
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=22,
        frameon=False,
        ncol=len(db_list),
        labelspacing=0.2,
        columnspacing=1.2,
        handlelength=1.5,
        borderaxespad=0,
        borderpad=0,
        handletextpad=0.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for the legend
    fig.subplots_adjust(hspace=0.16, wspace=0.10)
    
    output_filename = "param_combined.pdf"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"Result saved to {os.path.abspath(output_filename)}")