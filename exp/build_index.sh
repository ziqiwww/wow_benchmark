# run build_index.sh to build the index for the given data
project_path="../../"
# when the number of point in a single attribute subgraph is low (in extreme cases, 1), we can set w to a higher value and local_m to a lower one to avoid the overhead of searching in the attribute subgraph and that will change M

db="sift1m"
vec_type="fvecs"
attr_type="int"
base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"
space="l2"

# pre-allocate adequate memory for layers. as 2 * 4^10 > 1m > 2 * 4^9, we set wp=10
wp=10
# maximum out-degree of a node in a single layer
M=16
# used as window boost base o
l_M=4
# beam search width for construction \omega_c
efc=128

# ordered insert 1, unordered insert 0
is_ordered=0

# if is_ordered use ./buildspatt, else use ./buildspattplus
if [ $is_ordered -eq 1 ]; then
    executable="buildspatt"
else
    executable="buildspattplus"
fi

threads=16

index_path=$project_path"exp/index/wow"

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin

echo "Building index for $db", executable: $executable
echo "index saving prefix: " $index_path
# ./buildindex -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -ba "$base_attr" -s "$space" -o "$index_path" -dc "$decay" -m "$M" -lm "$l_M" -efc "$efc" -w "$w" -wp "$wp" -t "$threads"
./$executable -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -s "$space" -o "$index_path" -m "$M" -lm "$l_M" -efc "$efc" -wp "$wp" -t "$threads" # & export pid=$!
echo "Index building for $db with pid=$pid"
wait $pid



# cmd: ./buildindex -db sift10k -vt fvecs -at int -bv ../../exp/data/vecs/sift10k/sift10k_base.fvecs -ba ../../exp/data/meta/meta_int_10000_10000.bin -s l2 -o ../../exp/index -dc d_lin -m 64 -lm 4 -efc 100 -w 50 -p p_oneround -t 1
echo "Index built for $db"
