project_path="../../"

db="sift1m"
vec_type="fvecs"
attr_type="int"
nb=1000000
nc=1000000
k=10
space="l2"
base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"

query_attr_prefix=$project_path"exp/data/ranges/intrng10k_1000000/"
# gt=$project_path"exp/gt/ordered_sift1m_int_nb1000000_nc1000000_f0_k100_gt.ivecs"
index=$project_path"exp/index/hnsw/"
post_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"prefiltering_correlation_"
pre_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"prefiltering_correlation_"

cor_level="low"
M=16
efc=128
thread=12

query_vec=$project_path"build/bin/"$cor_level"_cor_vec.fvecs"
query_attr=$project_path"build/bin/"$cor_level"_cor_range.bin"
gt=$gt_prefix$frac"_k"$k"_gt.ivecs"

# fracs=(0 1 2 3 4 5 6 7 8 9 10)
# fracs=(5)

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin


post_result=$post_result_prefix"k"$k".csv"
pre_result=$pre_result_prefix"k"$k".csv"
./runnaive -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -qv "$query_vec" -qa "$query_attr" -gt "$gt" -i "$index" -o "$post_result" -e "$pre_result" -t "$thread" -k "$k" -s "$space" -m "$M" -efc "$efc"
echo "result saved to "$result
