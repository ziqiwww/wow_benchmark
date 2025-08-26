project_path="../../"

# dataset name
db="sift"
# vector type
vec_type="fvecs"
# attribute type
attr_type="int"
# number of base vecs
nb=1000000
# k for knn
k=10
# supported distance functions: l2, ip, cos
space="l2"
# base vec file
base_vec=$project_path"exp/data/vecs/sift1m/sift_base.fvecs"
# query vec file
query_vec=$project_path"exp/data/vecs/sift1m/sift_query.fvecs"
# query range file. with nq [x,y] pairs
query_attr_prefix=$project_path"exp/data/ranges/intrng10k_1000000/"
# index path, pp for unordered built index.
index_path=$project_path"exp/index/wow/pp_sift1m_fvecs_int_128_1000000_10_16_4.index"

# result path should be in the form of: ../../exp/result/sift10k_fvecs_int_128_10000_30_64_2_6273.csv generated from index_path
base_name=$(basename $index_path)

# dynamic layer range, 1 to use the selectivity-aware landing layer selection, 0 to try all posible landing layers for benchmarking
dl=1
dl_str="dy"
if [ $dl -eq 0 ]; then
    dl_str="st"
fi

# single thread benchmark
threads=1

# batch searching
# search using different efs and fracs
query_frac=(17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)

# compile
cd ..
mkdir -p build && cd build
cmake .. && make -j
cd ../build/bin

# if index starts with "layered", use searchspatt, else use searchspattplus
if [[ $index_path == *"layered"* ]]; then
    executable="searchspatt"
else
# assert index start with pp
    if [[ $index_path != *"pp"* ]]; then
        echo "Index path should start with pp"
        exit 1
    fi
    executable="searchspattplus"
fi


for frac in "${query_frac[@]}"
do
    query_attr=$query_attr_prefix$frac".bin"
    gt_path=$project_path"exp/data/gt/gt_"$db"_k"$k"/"$frac".bin"

    # generate ground truth at the first run
    if [ ! -f "$gt_path" ]; then
        echo "Generating ground truth for $db"
        ./gtmanager -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -qv "$query_vec" -qa "$query_attr" -gt "$gt_path" -k "$k" -s "$space"
        echo "Ground truth generated for $db"
    fi

    # make dir wow if not exists
    if [ ! -d "$project_path"exp/result/wow ]; then
        mkdir -p "$project_path"exp/result/wow
    fi

    result_dir=$project_path"exp/result/wow/"${base_name%.*}"_"${dl_str}"_k"$k
    # make dir if not exists
    if [ ! -d "$result_dir" ]; then
        mkdir -p "$result_dir"
    fi
    result_path=$result_dir"/"$frac".csv"
    echo "Result path: $result_path"
    
    # remove result if exists
    if [ -f "$result_path" ]; then
        rm "$result_path"
    fi

    echo "Searching for $db, executable: $executable"
    # ./search -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -ba "$base_attr" -qv "$query_vec" -qa "$query_attr" -gt "$gt_path" -i "$index_path" -o "$result_path" -k "$k" -s "$space" -efs "$efs" -t "$threads"
    # for ef in "${efs[@]}"
    # do
    ./$executable -db "$db" -vt "$vec_type" -at "$attr_type" -bv "$base_vec" -qv "$query_vec" -qa "$query_attr" -gt "$gt_path" -i "$index_path" -o "$result_path" -k "$k" -s "$space" -dl "$dl" -t "$threads" # & export pid=$!
    echo "Searching for $db with ef=$ef with pid=$pid"
    wait $pid
    # cmd: ./search -db sift10k -vt fvecs -at int -bv ../../exp/data/vecs/sift10k/sift10k_base.fvecs -ba ../../exp/data/meta/meta_int_10000_10000.bin -qv ../../exp/data/vecs/sift10k/sift10k_query.fvecs -qa ../../exp/data/ranges/intrng100/5.bin -gt ../../exp/gt/sift10k_int_10000_10000_gt.ivecs -i ../../exp/index/sift10k_fvecs_int_128_10000_30_64_2_6273.index -o ../../exp/result/sift10k_fvecs_int_128_10000_50_64_4_6273.csv -k 10 -s l2 -efs 50 -t 1
    echo "Search done for $db with ef=$ef"
    # done
    # cmd: ./search -db sift10k -vt fvecs -at int -bv ../../exp/data/vecs/sift10k/sift10k_base.fvecs -ba ../../exp/data/meta/meta_int_10000_10000.bin -qv ../../exp/data/vecs/sift10k/sift10k_query.fvecs -qa ../../exp/data/ranges/intrng100/5.bin -gt ../../exp/gt/sift10k_int_10000_10000_gt.ivecs -i ../../exp/index/sift10k_fvecs_int_128_10000_50_64_4_6273.index -o ../../exp/result/sift10k_fvecs_int_128_10000_50_64_4_6273.csv -k 10 -s l2 -efs 50 -t 1
    echo "Search done for $db, frac=$frac"
    echo "result saved to "$result_path
    # sleep 1s, avoid os post processing
    sleep 1s
done


