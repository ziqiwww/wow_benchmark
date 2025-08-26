perf_exec=/usr/lib/linux-tools/5.15.0-136-generic/perf
# usage: ./run-perf.sh <pid> <duration /s>, result: perf.svg
perf_dir=./perf
cd "$perf_dir"
# get pid of the program, if there are multiple processes with the same name, it will return the first one
pid=$1

# run below command for 5 seconds and signal the process to stop: perf_exec -e cpu-clock -g -p $1
echo "Running perf for $1, pid=$pid"
timeout -s SIGINT "$2"s $perf_exec record -e cpu-clock -g -p $pid
echo "perf done for $1, generating flamegraph"
$perf_exec script -i perf.data &> perf.unfold
./FlameGraph/stackcollapse-perf.pl perf.unfold &> perf.folded
./FlameGraph/flamegraph.pl perf.folded > perf.svg