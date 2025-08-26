#include <omp.h>
#include "hnswlib/hnswalg.h"
#include "common/dataloader.hh"
#include "common/type.hh"
#include "common/result.hh"
#include "apps/gtmanager.hh"
#include "argparse/argparse.hpp"

std::unordered_map<std::string, std::unique_ptr<hnswlib::HierarchicalNSW<spatt::dist_t>>> opened_hnsws;

template <typename vec_t = float>
auto GetHNSW(spatt::DataLoader<vec_t> &dl, int range_l, int range_r, hnswlib::SpaceInterface<vec_t> *space,
    const argparse::ArgumentParser &program) -> hnswlib::HierarchicalNSW<spatt::dist_t> *
{
  // if index file exists, load it
  std::string index_name = program.get<std::string>("database") + "_" + program.get<std::string>("vec-type") + "_" +
                           +"_" + program.get<std::string>("space") + "_" + std::to_string(program.get<size_t>("M")) +
                           "_" + std::to_string(range_l) + "_" + std::to_string(range_r) + ".hnsw";
  index_name = program.get<std::string>("index") + index_name;

  if (opened_hnsws.contains(index_name)) {
    return opened_hnsws[index_name].get();
  }
  size_t nb = range_r - range_l + 1;
  // M0 is 2 * M, so we need to divide it by 2
  auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<spatt::dist_t>>(
      space, nb, program.get<size_t>("M") / 2, program.get<size_t>("ef-construction"));
  if (std::filesystem::exists(index_name)) {
    hnsw->loadIndex(index_name, space, nb);
    // calculate the everage degree
    size_t total_degree = 0;
    for (size_t i = 0; i < hnsw->cur_element_count; ++i) {
      total_degree += hnsw->get_linklist0(i)[0];
    }
    LOG(fmt::format("Average degree: {}", total_degree / hnsw->cur_element_count));
    LOG("Index loaded");
  } else {
    LOG("Start building HNSW index");
    std::atomic<size_t> counter{0};
#pragma omp parallel for num_threads(program.get<int>("thread")) schedule(dynamic) shared(counter, dl, hnsw)
    for (int ib = range_l; ib <= range_r; ++ib) {
      auto v = dl.GetBaseVecByID(ib);
      hnsw->addPoint(v, ib);
      if (++counter % 1000 == 0) {
        LOG(fmt::format("Processed: {}/{}", counter.load(), dl.nb_));
      }
    }
    LOG("HNSW index building finished");
    // save
    hnsw->saveIndex(index_name);
  }
  opened_hnsws[index_name] = std::move(hnsw);
  return opened_hnsws[index_name].get();
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser program("runhnsw");
  program.add_argument("-db", "--database").help("Database name, e.g.: sift1m, sift1b").required();
  program.add_argument("-vt", "--vec-type").help("Vector type, supported types: fvecs, bvecs, ivecs").required();
  program.add_argument("-bv", "--base-vec").help("Base vector path").required();
  program.add_argument("-qv", "--query-vec").help("Query vector path").required();
  program.add_argument("-qa", "--query-attr").help("Query window range path").required();
  program.add_argument("-m", "--M").help("M value").default_value(16).scan<'u', size_t>();
  program.add_argument("-efc", "--ef-construction").help("Construction ef").default_value(128).scan<'u', size_t>();
  program.add_argument("-gt", "--ground-truth").help("Ground truth path").required();
  program.add_argument("-s", "--space").help("Space type, supported spaces: l2, ip").required();
  program.add_argument("-k", "--k").help("Top k").default_value(10).scan<'u', size_t>();
  program.add_argument("-i", "--index").help("Index file path").required();
  program.add_argument("-o", "--output").help("Output result prefix").required();
  program.add_argument("-e", "--pre_output").help("Prefiltering output result prefix").required();
  program.add_argument("-t", "--thread").help("Number of threads").default_value(1).scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    std::abort();
  }

  std::vector<size_t>      efs = {1700,
           1400,
           1100,
           1000,
           900,
           800,
           700,
           600,
           500,
           400,
           300,
           250,
           200,
           180,
           160,
           140,
           120,
           100,
           90,
           80,
           70,
           60,
           55,
           50,
           45,
           40,
           35,
           30,
           25,
           20,
           15,
           10};
  spatt::DataLoader<float> dl(program.get<std::string>("database"), program.get<std::string>("vec-type"));
  dl.Loadfves(program.get<std::string>("base-vec"));
  dl.Loadfves(program.get<std::string>("query-vec"), true);
  dl.LoadQueryFilter(program.get<std::string>("query-attr"));
  spatt::GTManager<float> gt_manager(
      &dl, program.get<std::string>("ground-truth"), program.get<size_t>("k"), program.get<std::string>("space"));
  gt_manager.LoadGT();

  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  if (program.get<std::string>("space") == "l2") {
    space = std::make_unique<hnswlib::L2Space>(dl.d_);
  } else if (program.get<std::string>("space") == "ip" || program.get<std::string>("space") == "cos") {
    space = std::make_unique<hnswlib::InnerProductSpace>(dl.d_);
  } else {
    LOG("Unsupported space");
    std::abort();
  }
  // run prefiltering
  LOG("Start prefiltering");
  auto   start     = std::chrono::high_resolution_clock::now();
  double tot_time  = 0;
  int    num_query = 1000;
  for (int i = 0; i < num_query; ++i) {
    auto                             query = dl.GetQueryVecByID(i);
    auto                             rng   = dl.GetQueryFilterByID(i);
    std::vector<spatt::dist_id_pair> result;
    auto                             start = std::chrono::high_resolution_clock::now();
    for (int j = rng.l_; j <= rng.u_; ++j) {
      auto dist = space->get_dist_func()(query, dl.GetBaseVecByID(j), space->get_dist_func_param());
      PUSH_HEAP(result, dist, j);
      if (result.size() > program.get<size_t>("k")) {
        POP_HEAP(result);
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    tot_time += std::chrono::duration<double>(end - start).count();
  }
  LOG(fmt::format("Prefiltering time: {}", tot_time));
  std::ofstream ofs(program.get<std::string>("pre_output"));
  ofs << fmt::format("1,{}", num_query / tot_time);
  ofs.close();

  // run postfiltering by hnsw
  LOG("Start postfiltering");
  auto hnsw = GetHNSW(dl, 0, dl.nb_ - 1, space.get(), program);
  for (auto efs : efs) {
    spatt::ResultAnalyser ra(gt_manager.GetGT());
    for (size_t i = 0; i < dl.nq_; ++i) {
      auto query                         = dl.GetQueryVecByID(i);
      auto rng                           = dl.GetQueryFilterByID(i);
      hnsw->metric_distance_computations = 0;
      hnsw->metric_hops                  = 0;
      hnsw->setEf(efs);
      auto                             start      = std::chrono::high_resolution_clock::now();
      auto                             result_que = hnsw->searchKnn(query, 3 * program.get<size_t>("k"));
      auto                             end        = std::chrono::high_resolution_clock::now();
      std::vector<spatt::dist_id_pair> result;
      result.reserve(result_que.size());
      while (!result_que.empty()) {
        auto [dist, id] = result_que.top();
        if (id >= rng.l_ && id <= rng.u_)
          result.emplace_back(dist, id);
        result_que.pop();
      }
      if (result.size() < program.get<size_t>("k")) {
        while (result.size() < program.get<size_t>("k")) {
          result.emplace_back(0, dl.nb_);
        }
      }
      ra.Step(std::chrono::duration<double>(end - start).count(),
          hnsw->metric_distance_computations,
          hnsw->metric_hops,
          ra.CalculateRecall(i, program.get<size_t>("k"), result));
    }
    ra.Finalize();
    ra.Dump(program.get<std::string>("output"), {false, efs, program.get<size_t>("k"), {0, 0}});
  }

  LOG(fmt::format("Naive done, post filter result dumped to {}, prefiltering qps: {}",
      program.get<std::string>("output"),
      num_query / tot_time));
}
