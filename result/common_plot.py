import numpy as np

EVAL_METRIC = "qps-recall"
# EVAL_METRIC="dist-recall"

RECALL_THRESHOLD = 0.85
METHOD_LIST = []
WST = [
    # "prefiltering",
    # "postfiltering",
    # "vamana-tree",
    # "optimized-postfiltering",
    # "smart-combined",
    # "three-split",
    "super-postfiltering",
]
IRANGE = [
    "irange-16",
    "irange-32",
]

HSIG = [
    "hsig-16",
]
SERF = ["serf"]
ACORN = ["acorn"]
MILVUS = ["milvus-hnsw"]

DIGRA=[
    "digra-static"
]
RANGE_PQ=[
    "rangepq"
]
OURS_LAYERED = [
    # "spatt-layered-7-16-8-dy",
    # "spatt-layered-9-16-6-dy",
    "spatt-layered-10-16-4-dy",
    "spatt-layered-11-16-4-dy",
    "spatt-layered-12-16-4-dy",
    # "spatt-layered-20-16-2-dy",
    # "spatt-layered-11-32-4",
    # "spatt-layered-17-16-2",
]
OURS_PP = [
    # "spatt-pp-7-16-8-dy",
    # "spatt-pp-9-16-6-dy",
    "spatt-pp-circle-10-16-4-dy",
    "spatt-pp-10-16-4-dy",
    "spatt-pp-11-16-4-dy",
    "spatt-pp-12-16-4-dy",
    # "spatt-pp-10-16-4",
    # "spatt-pp-20-16-2-dy",
]

STATIC_INDEX = []
STATIC_INDEX.extend(WST)
STATIC_INDEX.extend(IRANGE)

INORDER_INDEX = []
INORDER_INDEX.extend(SERF)
INORDER_INDEX.extend(OURS_LAYERED)

OUTOFORDER_INDEX = []
OUTOFORDER_INDEX.extend(OURS_PP)
OUTOFORDER_INDEX.extend(HSIG)
OUTOFORDER_INDEX.extend(ACORN)
OUTOFORDER_INDEX.extend(MILVUS)

POSTINC_INDEX= []
POSTINC_INDEX.extend(DIGRA)
POSTINC_INDEX.extend(RANGE_PQ)

OUTOFORDER_INDEX.extend(["postfiltering", "prefiltering"])

COLOR_LIST = [
    "#FF3333",
    "#0099FF",
    "#33CC33",
    "#999900",
    "#FFCC00",
    "#C66537",
    "#CC66CC",
    "#808080",
]

# marker : https://matplotlib.org/stable/api/markers_api.html
# linestyle : https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html


def get_line_style(method):
    if method == "postfiltering" or method == "oracle_hnsw":
        return "solid"
    if method in STATIC_INDEX:
        return "dotted"
    elif method in INORDER_INDEX:
        return "dashdot"
    elif method in OUTOFORDER_INDEX:
        return "solid"
    elif method in POSTINC_INDEX:
        return "dashed"
    else:
        raise ValueError(f"method {method} not found")


def get_marker(method):
    if method == "postfiltering" or method == "oracle_hnsw":
        return "*"
    elif method in WST:
        return "^"
    elif method in IRANGE:
        return "X"
    elif method in SERF:
        return "d"
    elif method in OURS_LAYERED or method in OURS_PP:
        return "o"
    elif method in HSIG:
        return "s"
    elif method in ACORN:
        return "p"
    elif method in MILVUS:
        return "v"
    elif method in DIGRA:
        return "P"
    elif method in RANGE_PQ:
        # possible markers: "8", "<", ">", "1", "2", "3", "4", "|", "_"
        return ">"
    else:
        raise ValueError(f"method {method} not found")


def get_line_color(method):
    if method == "oracle_hnsw":
        return "#A8A8A8"
    elif method in OURS_LAYERED:
        return "#0099FF"
    elif method in OURS_PP:
        if method == "spatt-pp-circle-10-16-4-dy":
            return "#FFA500"
        return "#FF3333" #FF6600"
    elif method == "postfiltering":
        return "#A8A8A8"
    elif method == "prefiltering":
        return "#FFB74D" 
    elif method in WST:
        return "#C66537"
    elif method in IRANGE:
        return "#33CC33"
    elif method in SERF:
        return "#CC66CC"
    elif method in HSIG:
        return "#999900"
    elif method in ACORN:
        return "#2e8a5c"
    elif method in MILVUS:
        return "#4747C2"
    elif method in DIGRA:
        return "#003f5c"
    elif method in RANGE_PQ:
        return "#b6042a"
    else:
        raise ValueError(f"method {method} not found")


def get_x_y_data_with_metric(data, eval_metric):
    # if data only has 3 or less columns, return all 0 for y
    if data.shape[1] <= 3 and eval_metric == "dist-recall":
        return np.zeros_like(data[:, 1]), np.zeros_like(data[:, 1])
    if eval_metric == "qps-recall":
        return data[:, 1], data[:, 2]
    elif eval_metric == "dist-recall":
        return data[:, 1], data[:, 3]


def get_x_y_label_with_metric(eval_metric, k):
    if eval_metric == "qps-recall":
        return f"Recall@{k}", "QPS"
    elif eval_metric == "dist-recall":
        return f"Recall@{k}", "DC"


def get_db_name_beautify(db_name):
    if db_name == "arxiv":
        return "ArXiv"
    elif db_name == "gist":
        return "Gist"
    elif db_name == "wit":
        return "WIT"
    elif db_name == "sift":
        return "Sift"
    elif db_name == "wiki4m":
        return "Wikidata4M"
    elif db_name == "deep10m":
        return "Deep10M"
    else:
        return db_name


def get_method_beautify(method):
    if method == "oracle_hnsw":
        return "Oracle HNSW"
    elif method in WST:
        return "WST"
    elif method in IRANGE:
        return "iRangeGraph"
    elif method in SERF:
        return "SeRF"
    elif method in OURS_LAYERED:
        return "WoW (ordered)"
    elif method in OURS_PP:
        if method == "spatt-pp-circle-10-16-4-dy":
            return "WoW (circular)"
        return "WoW"
    elif method in HSIG:
        return "HSIG"
    elif method == "postfiltering":
        return "Post-filter"
    elif method == "prefiltering":
        return "Pre-filter"
    elif method == "acorn":
        return "ACORN"
    elif method == "milvus-hnsw":
        return "Milvus"
    elif method in DIGRA:
        if method == "digra-static":
            return "DIGRA"
    elif method in RANGE_PQ:
        return "RangePQ"
    else:
        raise ValueError(f"method {method} not found")


def get_z_order(method):
    z_order_list = ["postfiltering", WST, HSIG, ACORN,  SERF, IRANGE, DIGRA, DSG, MILVUS, OURS_LAYERED, OURS_PP, "oracle_hnsw", "RANGE_PQ", "prefiltering",]
    # use a list to define the z_order
    for i, m in enumerate(z_order_list):
        if method == m or method in m:
            return i


from matplotlib.ticker import FuncFormatter


# Define the custom transformation function
def custom_transform(y):
    return np.log2(y)


def custom_inverse_transform(y):
    return 2**y


# y_ticks = [
#     10,
#     20,
#     40,
#     60,
#     100,
#     200,
#     400,
#     600,
#     1000,
#     2000,
#     4000,
#     6000,
#     10000,
#     20000,
#     40000,
#     60000,
#     100000,
#     200000,
#     400000,
#     600000,
#     1000000,
#     2000000,
# ]
# y_ticks_labels = [
#     "$1\\times10^1$",
#     "$2\\times10^1$",
#     "$4\\times10^1$",
#     "$6\\times10^1$",
#     "$1\\times10^2$",
#     "$2\\times10^2$",
#     "$4\\times10^2$",
#     "$6\\times10^2$",
#     "$1\\times10^3$",
#     "$2\\times10^3$",
#     "$4\\times10^3$",
#     "$6\\times10^3$",
#     "$1\\times10^4$",
#     "$2\\times10^4$",
#     "$4\\times10^4$",
#     "$6\\times10^4$",
#     "$1\\times10^5$",
#     "$2\\times10^5$",
#     "$4\\times10^5$",
#     "$6\\times10^5$",
#     "$1\\times10^6$",
#     "$2\\times10^6$",
# ]

# y_ticks with 10, 30, 60, 100, 300, 600, 
# y_ticks = [
#     10,
#     30,
#     60,
#     100,
#     300,
#     600,
#     1000,
#     3000,
#     6000,
#     10000,
#     30000,
#     60000,
#     100000,
#     300000,
#     600000,
#     1000000,
# ]

# y_ticks_labels = [
#     "$1\\times10^1$",
#     "$3\\times10^1$",
#     "$6\\times10^1$",
#     "$1\\times10^2$",
#     "$3\\times10^2$",
#     "$6\\times10^2$",
#     "$1\\times10^3$",
#     "$3\\times10^3$",
#     "$6\\times10^3$",
#     "$1\\times10^4$",
#     "$3\\times10^4$",
#     "$6\\times10^4$",
#     "$1\\times10^5$",
#     "$3\\times10^5$",
#     "$6\\times10^5$",
#     "$1\\times10^6$",
# ]

# y_ticks with 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000
y_ticks = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
]
y_ticks_labels = [
    "$1\\times10^1$",
    "$2\\times10^1$",
    "$5\\times10^1$",
    "$1\\times10^2$",
    "$2\\times10^2$",
    "$5\\times10^2$",
    "$1\\times10^3$",
    "$2\\times10^3$",
    "$5\\times10^3$",
    "$1\\times10^4$",
    "$2\\times10^4$",
    "$5\\times10^4$",
    "$1\\times10^5$",
    "$2\\times10^5$",
    "$5\\times10^5$",
    "$1\\times10^6$",
]

# yticks with 10, 50,100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
# y_ticks = [
#     10,
#     50,
#     100,
#     500,
#     1000,
#     5000,
#     10000,
#     50000,
#     100000,
#     500000,
#     1000000,
# ]
# y_ticks_labels = [
#     "$1\\times10^1$",
#     "$5\\times10^1$",
#     "$1\\times10^2$",
#     "$5\\times10^2$",
#     "$1\\times10^3$",
#     "$5\\times10^3$",
#     "$1\\times10^4$",
#     "$5\\times10^4$",
#     "$1\\times10^5$",
#     "$5\\times10^5$",
#     "$1\\times10^6$",
# ]


def get_y_ticks_ylabels(y_data, use_lower_bound):
    # y_max = np.max(y_data)
    # if y_max < 1000:
    #     return [ 200, 400, 600, 800, 1000, 2000], ["$2\\times10^2$", "$4\\times10^2$", "$6\\times10^2$", "$8\\times10^2$", "$1\\times10^3$", "$2\\times10^3$"]
    # elif y_max < 2000:
    #     return [400, 800, 1200, 1600, 2000, 4000], ["0", "$4\\times10^2$", "$8\\times10^2$", "$1.2\\times10^3$", "$1.6\\times10^3$", "$2\\times10^3$", "$4\\times10^3$"]
    # ...

    y_max = np.max(y_data)
    y_min = np.min(y_data)
    y_min_pos = 0
    upper_margin = 2
    if use_lower_bound:
        upper_margin = 1
        y_min_pos = len(y_ticks) - 1
        for i in range(len(y_ticks) - 1, -1, -1):
            if y_min > y_ticks[i]:
                y_min_pos = i
                break
    for i in range(len(y_ticks) - 1):
        if y_max < y_ticks[i]:
            return y_ticks[y_min_pos : i + upper_margin], y_ticks_labels[y_min_pos : i + upper_margin]


METHOD_LIST.extend(OURS_PP)
METHOD_LIST.extend(OURS_LAYERED)
METHOD_LIST.extend(SERF)
METHOD_LIST.extend(WST)
METHOD_LIST.extend(IRANGE)
METHOD_LIST.extend(HSIG)
# METHOD_LIST.extend(DSG)
METHOD_LIST.extend(RANGE_PQ)
METHOD_LIST.extend(DIGRA)
METHOD_LIST.extend(MILVUS)
METHOD_LIST.extend(ACORN)

METHOD_LIST.append("postfiltering")
METHOD_LIST.append("prefiltering")

# generate method-color mapping should make sure the colors are different and distinguishable

RANGE_FRACTIONS = range(0, 18)
# RANGE_FRACTIONS = [17, 2, 5, 8, 11]
# RANGE_FRACTIONS = [17, 0, 3, 6, 9]
RANGE_FRACTIONS = [17, 1, 4, 7, 10]
