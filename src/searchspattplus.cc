#include "spattplus/spattplussearcher.hh"
#include "argparse/argparse.hpp"
#include "common/result.hh"

template <typename vec_t = float>
void RunSearching(const argparse::ArgumentParser &program, spatt::SearchParameters sp, spatt::DataLoader<vec_t> &dl,
    spatt::GTManager<vec_t> &gt_manager, spatt::SpattPlusSearcher<vec_t> &searcher)
{
  spatt::ResultAnalyser ra(gt_manager.GetGT());
  LOG("Start searching index");
  for (size_t i = 0; i < 1000; ++i) {
    // for(size_t i = 0; i < 1; ++i) {
    auto query = dl.GetQueryVecByID(i);
    auto rng   = dl.GetQueryFilterByID(i);
    auto ss = searcher.Search(query, rng, sp);
    std::sort(ss.result_.begin(), ss.result_.end());
    ra.Step(ss.rt_status_.run_time_,
        ss.rt_status_.dist_computation_,
        ss.rt_status_.hop_,
        ra.CalculateRecall(i, program.get<size_t>("k"), ss.result_));
  }
  ra.Finalize();
  ra.Dump(program.get<std::string>("output"), sp);
}

int main(int args, char *argv[])
{
  argparse::ArgumentParser program("searchindex");
  program.add_argument("-db", "--database").help("Database name, e.g.: sift1m, sift1b").required();
  program.add_argument("-vt", "--vec-type").help("Vector type, supported types: fvecs, bvecs, ivecs").required();
  program.add_argument("-at", "--attr-type").default_value("label").help("Currently only support vector id");
  program.add_argument("-bv", "--base-vec").help("Base vector path").required();
  program.add_argument("-qv", "--query-vec").help("Query vector path").required();
  program.add_argument("-qa", "--query-attr").help("Query window range path").required();
  program.add_argument("-gt", "--ground-truth").help("Ground truth path").required();
  program.add_argument("-i", "--index").help("Index file path").required();
  program.add_argument("-o", "--output").help("Output result prefix").required();
  program.add_argument("-s", "--space").help("Space type, supported spaces: l2, ip, cos").required();
  // program.add_argument("-efs", "--ef-search").help("Search ef").default_value(100).scan<'u', size_t>();
  program.add_argument("-k", "--k").help("Top k").default_value(10).scan<'u', size_t>();
  program.add_argument("-dl", "--dynamic-layer").help("Dynamic layer").default_value(1).scan<'i', int>();
  program.add_argument("-t", "--thread").help("Number of threads").default_value(1).scan<'i', int>();

  try {
    program.parse_args(args, argv);
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
  // std::vector<size_t> efs = {100};
  // check vec format and create dataloader
  spatt::DataLoader<float> dl(program.get<std::string>("database"),
      program.get<std::string>("vec-type"),
      program.get<std::string>("attr-type"));
  spatt::GTManager<float>  gt_manager(
      &dl, program.get<std::string>("ground-truth"), program.get<size_t>("k"), program.get<std::string>("space"));
  gt_manager.LoadGT();
  dl.Loadfves(program.get<std::string>("base-vec"));
  dl.Loadfves(program.get<std::string>("query-vec"), true);
  dl.LoadQueryFilter(program.get<std::string>("query-attr"));
  spatt::SpattPlusSearcher<float> searcher(&dl, program.get<std::string>("space"));
  searcher.LoadIndex(program.get<std::string>("index"));
  std::vector<int> layer_low_vec;
  layer_low_vec.reserve(searcher.iparam_.wp_ + 1);
  for (int i = 0; i <= searcher.iparam_.wp_; ++i) {
    layer_low_vec.emplace_back(i);
  }
  for (auto ef : efs) {
    if (program.get<int>("dynamic-layer") == 1) {
      LOG("Searching with dynamic layer decision");
      spatt::SearchParameters sp = {.is_dynamic_ = true,
          .efs_                                  = ef,
          .k_                                    = program.get<size_t>("k"),
          .layer_rng_                            = {static_cast<int>(searcher.iparam_.wp_ + 1), -1}};
      RunSearching(program, sp, dl, gt_manager, searcher);
    } else {
      LOG("Searching with layer range full scan");
      for (auto layer_low : layer_low_vec) {
        spatt::SearchParameters sp = {
            .is_dynamic_ = false, .efs_ = ef, .k_ = program.get<size_t>("k"), .layer_rng_ = {0, layer_low}};
        RunSearching(program, sp, dl, gt_manager, searcher);
      }
    }
  }
}