project_path="../../"

db="sift1m"
vec_type="fvecs"
attr_type="int"
nb=1000000
nc=1000000
k=10
space="l2"
base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"
query_vec=$project_path"exp/data/vecs/sift1m/sift_query.fvecs"
query_attr_prefix=$project_path"exp/data/ranges/intrng10k_1000000/"
gt_prefix=$project_path"exp/gt/ordered_"$db"_"$attr_type"_nb"$nb"_nc"$nc"_f"
# gt=$project_path"exp/gt/ordered_sift1m_int_nb1000000_nc1000000_f0_k100_gt.ivecs"
index=$project_path"exp/index/hnsw/"
post_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"postfiltering_f"
pre_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"prefiltering_f"

M=16
efc=128
thread=12

fracs=(0 1 2 3 4 5 6 7 8 9 10)
fracs=(2)

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin

for frac in "${fracs[@]}"
do
    query_attr=$query_attr_prefix$frac".bin"
    gt=$gt_prefix$frac"_k"$k"_gt.ivecs"
    post_result=$post_result_prefix$frac"_k"$k".csv"
    pre_result=$pre_result_prefix$frac"_k"$k".csv"
    ./gen_corquery -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -qv "$query_vec" -qa "$query_attr" -gt "$gt" -i "$index" -o "$post_result" -e "$pre_result" -t "$thread" -k "$k" -s "$space" -m "$M" -efc "$efc"
    echo "result saved to "$result
done