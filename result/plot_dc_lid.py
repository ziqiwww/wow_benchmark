import os
import numpy as np
import matplotlib.pyplot as plt
from common_plot import get_db_name_beautify
from matplotlib.ticker import ScalarFormatter,FuncFormatter,FixedLocator


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['text.usetex'] = True

def get_lid_db(db, fraction):
    lid_csv = f"lid_{db}.csv"
    # csv is fraction,lid
    data = np.loadtxt(lid_csv, delimiter=",", ndmin=2)
    print(f"loading csv {lid_csv}")
    # Assumed CSV format: [fraction, lid]
    lid = data[data[:, 0] == fraction, 1]
    if lid.size == 0:
        raise ValueError(f"No data found for fraction {fraction} in {lid_csv}")
    # keep 2 decimals
    return round(lid[0], 2)

def get_lid_color(db):
    # Define a color map for LID values
    color_map = {
        "sift": "#79021c",  # red #e41a1c
        "gist": "#1e3d58",  # blue #377eb8 "#e41a1c", "#377eb8"
    }
    return color_map.get(db, "#000000")  # default to black if db not found

if __name__ == "__main__":
    color_list = ["#ffadad", "#a2d2ff", "#4daf4a", "#984ea3", "#ff7f00"]

    width = 4
    height = 2.5

    fig, axes = plt.subplots(1, 2, figsize=(width * 2, height), dpi=150)
    axes = axes.flatten()

    db_list = ["sift", "gist"]
    db_color_map = {db: color for db, color in zip(db_list, color_list)}
    
    target_efs = 1000
    K = 10
    fractions = [0, 1, 4, 7, 10]
    bar_width = 0.35
    x = np.arange(len(fractions))
    
    # Data storage
    dc_values = {db: [] for db in db_list}
    recall_values = {db: [] for db in db_list}
    lid_values = {db: [] for db in db_list}
    
    # Load data
    for db in db_list:
        for fraction in fractions:
            try:
                # Get LID values
                lid = get_lid_db(db, fraction)
                lid_values[db].append(lid)
                
                # Get DC and Recall values
                csv_dir = f"{db}_log_csv_k{K}/spatt-pp-10-16-4-dy"
                csv_file = f"{csv_dir}/{fraction}.csv"
                
                data = np.loadtxt(csv_file, delimiter=",", ndmin=2)
                print(f"loading csv {csv_file}")
                
                # Find row with target_efs
                row = data[data[:, 0] == target_efs]
                if row.size == 0:
                    print(f"Warning: No data for efs={target_efs} in {csv_file}")
                    dc_values[db].append(np.nan)
                    recall_values[db].append(np.nan)
                    continue
                
                # Extract values (assumed format: [num_threads, recall, qps, dc, ...])
                recall = row[0, 1]  # Recall is at index 1
                dc = row[0, 3]      # DC is at index 3
                
                dc_values[db].append(dc)
                recall_values[db].append(recall)
                
            except Exception as e:
                print(f"Error processing {db}, fraction {fraction}: {e}")
                dc_values[db].append(np.nan)
                recall_values[db].append(np.nan)
                lid_values[db].append(np.nan)
    
    # Plot DC values
    ax = axes[0]
    for i, db in enumerate(db_list):
        bars = ax.bar(x + (i - 0.5) * bar_width, dc_values[db], width=bar_width, 
                      color=db_color_map[db], label=get_db_name_beautify(db))
        
        # Add LID values inside the bars
        for j, bar in enumerate(bars):
            if not np.isnan(dc_values[db][j]) and not np.isnan(lid_values[db][j]):
                lid_text = f"{lid_values[db][j]:.2f}"
                y_bottom = ax.get_ylim()[0]  # Get the bottom limit
                y_pos_min = y_bottom + 170
                # text_y_pos should be in the middle of the bar and above the y_pos_min
                text_y_pos = max(y_pos_min, (bar.get_height() - y_bottom) / 2 + y_bottom - 1500)  # Small offset from bottom
                
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    text_y_pos,
                    lid_text,
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=14,
                    color=get_lid_color(db)  # Use LID color for text
                )
    
    ax.set_ylabel("DC", fontsize=18, labelpad=0.5)
    ax.set_xlabel("Range fraction", fontsize=18, labelpad=0.5)
    ax.set_xticks(x)
    x_tick_text = [r"$2^{-%d}$" % f for f in fractions]
    x_tick_text[0] = r"$2^{0}$"  # Change first tick to 2^0
    ax.set_xticklabels(x_tick_text, fontsize=18)
    # Create a legend with 2 columns and no frame
    legend = ax.legend(loc='upper right', frameon=False, handlelength=1, fontsize=16,handletextpad=0.5)
    ax.tick_params(axis='both', pad=0.4)
    ax.grid(False)
    
    # Plot Recall values
    ax = axes[1]
    # Set y-axis range for recall subplot to [0.93, 1]
    ax.set_ylim(0.93, 1.0)
    
    for i, db in enumerate(db_list):
        bars = ax.bar(x + (i - 0.5) * bar_width, recall_values[db], width=bar_width, 
                     color=db_color_map[db], label=get_db_name_beautify(db))
        
        # Add LID values inside the bars
        for j, bar in enumerate(bars):
            if not np.isnan(recall_values[db][j]) and not np.isnan(lid_values[db][j]):
                lid_text = f"{lid_values[db][j]:.2f}"
                
                # Calculate text position - place it just above the bottom of the visible area
                y_bottom = ax.get_ylim()[0]  # Get the bottom limit (0.93)
                y_pos_min = y_bottom + 0.001
                # text_y_pos should be in the middle of the bar and above the y_pos_min
                # print(f"y_bottom: {y_bottom}, bar height: {bar.get_height()}, y_pos_min: {y_pos_min}")
                text_y_pos = max(y_pos_min, (bar.get_height() - y_bottom) / 2 + y_bottom -0.008)  # Small offset from bottom
                
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    text_y_pos,
                    lid_text,
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=14,
                    color=get_lid_color(db)  # Use LID color for text
                )
    
    ax.set_ylabel(f"Recall@{K}", fontsize=18, labelpad=0)
    ax.set_xlabel("Range fraction", fontsize=18, labelpad=0)
    ax.set_xticks(x)
    x_tick_text = [r"$2^{-%d}$" % f for f in fractions]
    x_tick_text[0] = r"$2^{0}$"  # Change first tick to 2^0
    ax.set_xticklabels(x_tick_text, fontsize=18)
    ax.tick_params(axis='both', pad=0.4)
    ax.grid(False)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    
    output_filename = "dc_recall_lid.pdf"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"Result saved to {os.path.abspath(output_filename)}")