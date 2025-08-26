#include <omp.h>
#include "hnswlib/hnswalg.h"
#include "common/dataloader.hh"
#include "common/type.hh"
#include "common/result.hh"
#include "apps/gtmanager.hh"
#include "argparse/argparse.hpp"

// FIXME: this implementation does not reflect fair QPS result because one query will try all efs. compared to
// go through all queries within each single efs, this implementation will be more cache friendly.

std::unordered_map<std::string, std::unique_ptr<hnswlib::HierarchicalNSW<spatt::dist_t>>> opened_hnsws;

template <typename vec_t = float>
auto GetHNSW(spatt::DataLoader<vec_t> &dl, int range_l, int range_r, hnswlib::SpaceInterface<vec_t> *space,
    std::string &index_name, const argparse::ArgumentParser &program) -> hnswlib::HierarchicalNSW<spatt::dist_t> *
{
  // if index file exists, load it
  index_name = program.get<std::string>("database") + "_" + program.get<std::string>("vec-type") + "_" +
               program.get<std::string>("attr-type") + "_" + program.get<std::string>("space") + "_" +
               std::to_string(program.get<size_t>("M")) + "_" + std::to_string(range_l) + "_" +
               std::to_string(range_r) + ".hnsw";
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
    auto                start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(program.get<int>("thread")) schedule(dynamic) shared(counter, dl, hnsw)
    for (int ib = range_l; ib <= range_r; ++ib) {
      auto v = dl.GetBaseVecByID(ib);
      hnsw->addPoint(v, ib);
      if (++counter % 1000 == 0) {
        LOG(fmt::format("Processed: {}/{}", counter.load(), dl.nb_));
      }
    }
    LOG(fmt::format("HNSW index built in {}s",
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count()));
    // save
    if (range_r - range_l + 1 == dl.nb_) {
      hnsw->saveIndex(index_name);
    }
  }
  opened_hnsws[index_name] = std::move(hnsw);
  return opened_hnsws[index_name].get();
}

template <typename vec_t = float>
void RunHNSW(const argparse::ArgumentParser &program, spatt::DataLoader<vec_t> &dl,
    spatt::GTManager<vec_t> &gt_manager, std::vector<size_t> &efs)
{
  hnswlib::SpaceInterface<vec_t> *space = nullptr;
  if (program.get<std::string>("space") == "l2") {
    space = new hnswlib::L2Space(dl.d_);
  } else if (program.get<std::string>("space") == "ip") {
    space = new hnswlib::InnerProductSpace(dl.d_);
  } else {
    LOG(fmt::format("Unsupported space: {}", program.get<std::string>("space")));
    std::abort();
  }
  LOG("Start searching HNSW index");
  std::unordered_map<size_t, std::unique_ptr<spatt::ResultAnalyser>> result_analysers;
  for (auto ef : efs) {
    result_analysers[ef] = std::make_unique<spatt::ResultAnalyser>(gt_manager.GetGT());
  }
  int batch_size = 100;  // batch size is used to reduce the influcence of cache locality if we run a single query on
                         // all efs
  for (size_t i = 0; i < 1000; i += batch_size) {
    std::vector<std::string>                               index_names;
    std::vector<hnswlib::HierarchicalNSW<spatt::dist_t> *> hnsws;
    for (size_t j = 0; j < batch_size; j++) {
      std::string index_name;
      auto        range_filter = dl.GetQueryFilterByID(i + j);
      hnsws.push_back(GetHNSW(dl, range_filter.l_, range_filter.u_, space, index_name, program));
      index_names.push_back(index_name);
    }
    for (auto ef : efs) {
      spatt::ResultAnalyser &ra = *result_analysers[ef];
      for (size_t j = 0; j < batch_size; j++) {
        hnsws[j]->metric_distance_computations = 0;
        hnsws[j]->metric_hops                  = 0;
        hnsws[j]->setEf(ef);
        auto start      = std::chrono::high_resolution_clock::now();
        auto result_que = hnsws[j]->searchKnn(dl.GetQueryVecByID(i + j), program.get<size_t>("k"));
        auto end        = std::chrono::high_resolution_clock::now();
        std::vector<spatt::dist_id_pair> result;
        result.reserve(result_que.size());
        while (!result_que.empty()) {
          auto [dist, id] = result_que.top();
          result.emplace_back(dist, id);
          result_que.pop();
        }
        ra.Step(std::chrono::duration<double>(end - start).count(),
            hnsws[j]->metric_distance_computations,
            hnsws[j]->metric_hops,
            ra.CalculateRecall(i + j, program.get<size_t>("k"), result));
      }
    }
    // do clean ups
    for (int j = 0; j < batch_size; j++) {
      auto query_range = dl.GetQueryFilterByID(i + j);
      if (query_range.u_ - query_range.l_ + 1 == dl.nb_) {
        continue;
      }
      auto name = index_names[j];
      opened_hnsws.erase(name);
      if (std::filesystem::exists(name)) {
        std::filesystem::remove(name);
      }
    }
  }
  for (auto ef : efs) {
    auto &ra = *result_analysers[ef];
    ra.Finalize();
    ra.Dump(program.get<std::string>("output"), {false, ef, program.get<size_t>("k"), {0, 0}});
  }
  delete space;
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser program("gtmanager");
  program.add_argument("-db", "--database").help("Database name, e.g.: sift1m, sift1b").required();
  program.add_argument("-vt", "--vec-type").help("Vector type, supported types: fvecs, bvecs, ivecs").required();
  program.add_argument("-at", "--attr-type").help("Attribute type, supported types: int, float, timestamp").required();
  program.add_argument("-bv", "--base-vec").help("Base vector path").required();
  program.add_argument("-ba", "--base-attr").help("Base attribute path").required();
  program.add_argument("-qv", "--query-vec").help("Query vector path").required();
  program.add_argument("-qa", "--query-attr").help("Query window range path").required();
  program.add_argument("-m", "--M").help("M value").default_value(16).scan<'u', size_t>();
  program.add_argument("-efc", "--ef-construction").help("Construction ef").default_value(128).scan<'u', size_t>();
  program.add_argument("-gt", "--ground-truth").help("Ground truth path").required();
  program.add_argument("-s", "--space").help("Space type, supported spaces: l2, ip").required();
  program.add_argument("-k", "--k").help("Top k").default_value(10).scan<'u', size_t>();
  program.add_argument("-i", "--index").help("Index file path").required();
  program.add_argument("-o", "--output").help("Output result prefix").required();
  program.add_argument("-t", "--thread").help("Number of threads").default_value(1).scan<'i', int>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    std::abort();
  }

  std::vector<size_t> efs = {1700,
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

  spatt::DataLoader<float> dl(program.get<std::string>("database"),
      program.get<std::string>("vec-type"),
      program.get<std::string>("attr-type"));
  dl.Loadfves(program.get<std::string>("base-vec"));
  dl.Loadfves(program.get<std::string>("query-vec"), true);
  dl.LoadQueryFilter(program.get<std::string>("query-attr"));
  spatt::GTManager<float> gt_manager(
      &dl, program.get<std::string>("ground-truth"), program.get<size_t>("k"), program.get<std::string>("space"));
  gt_manager.LoadGT();
  RunHNSW(program, dl, gt_manager, efs);
  return 0;
}
