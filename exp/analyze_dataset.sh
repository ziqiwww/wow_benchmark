project_path="../../"

# db="sift"
# vec_type="fvecs"
# attr_type="int"
# nb=1000000
# nc=1000000
# k=10
# space="l2"
# base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"
# base_attr=$project_path"exp/data/meta/meta_"$attr_type"_"$nb"_"$nc".bin"
# query_vec=$project_path"exp/data/vecs/sift1m/sift_query.fvecs"
# query_attr_prefix=$project_path"exp/data/ranges/intrng10k_1000000/"


# db="arxiv"
# vec_type="fvecs"
# attr_type="int"
# nb=2138591
# nc=2138591
# k=10
# space="cos"
# base_vec=$project_path"exp/data/vecs/arxiv2m/arxiv2m_base.fvecs"
# base_attr=$project_path"exp/data/meta/meta_"$attr_type"_"$nb"_"$nc".bin"
# query_vec=$project_path"exp/data/vecs/arxiv2m/arxiv2m_query.fvecs"
# query_attr_prefix=$project_path"exp/data/ranges/intrng10k_2138591/"


# db="wiki4m"
# vec_type="fvecs"
# attr_type="int"
# nb=4000000
# nc=4000000
# k=10
# space="cos"
# base_vec=$project_path"exp/data/vecs/wiki/wikien4m_base.fvecs"
# base_attr=$project_path"exp/data/meta/meta_"$attr_type"_"$nb"_"$nc".bin"
# query_vec=$project_path"exp/data/vecs/wiki/wikien_query.fvecs"
# query_attr_prefix=$project_path"exp/data/ranges/intrng10k_4000000/"

db="deep10m"
vec_type="fvecs"
attr_type="int"
nb=10000000
nc=10000000
k=10
space="l2"
base_vec=$project_path"exp/data/vecs/deep1b/deep10m_base.fvecs"
base_attr=$project_path"exp/data/meta/meta_"$attr_type"_"$nb"_"$nc".bin"
query_vec=$project_path"exp/data/vecs/deep1b/deep1B_queries.fvecs"
query_attr_prefix=$project_path"exp/data/ranges/intrng10k_10000000/"



gt_prefix=$project_path"exp/data/gt/gt_"$db"_k"$k"/"

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin

./analyze_dataset -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -ba "$base_attr" -qv "$query_vec" -qa "$query_attr_prefix" -gt "$gt_prefix" -k "$k" -s "$space"