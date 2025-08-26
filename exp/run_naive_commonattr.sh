project_path="../../"

db="sift1m"
vec_type="fvecs"
attr_type="int"
nb=1000000
nc=100
k=10
space="l2"
base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"
query_vec=$project_path"exp/data/vecs/sift1m/sift_query.fvecs"
base_attr=$project_path"exp/data/meta/meta_c"$nc"_n"$nb".bin"
# gt=$project_path"exp/gt/ordered_sift1m_int_nb1000000_nc1000000_f0_k100_gt.ivecs"
index=$project_path"exp/index/hnsw/"
post_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"postfiltering_c"
pre_result_prefix=$project_path"exp/result/"$db"_"$vec_type"_"$attr_type"prefiltering_c"

M=16
efc=128
thread=12
multiplier=$nc

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin

query_attr=$project_path"exp/data/ranges/common_ranges_c"$nc"_nq10000.bin"
gt=$project_path"example/common_"$db"_gt_c"$nc"_nq10000.bin"
post_result=$post_result_prefix$nc"_k"$k".csv"
pre_result=$pre_result_prefix$nc"_k"$k".csv"
./runnaive_common -db "$db" -vt "$vec_type" -bv "$base_vec" -ba $base_attr -qv "$query_vec" -qa "$query_attr" -gt "$gt" -i "$index" -o "$post_result" -e "$pre_result" -t "$thread" -k "$k" -s "$space" -m "$M" -efc "$efc" -mul "$multiplier"
echo "result saved to "$result
