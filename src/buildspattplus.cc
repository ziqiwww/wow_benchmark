
#include "spattplus/spattplusbuilder.hh"
#include "argparse/argparse.hpp"

template <typename vec_t>
void RunBuilding(const argparse::ArgumentParser &program)
{
  // check vec format and create dataloader
  spatt::DataLoader<vec_t> dl(program.get<std::string>("database"),
      program.get<std::string>("vec-type"),
      program.get<std::string>("attr-type"));
  dl.Loadfves(program.get<std::string>("base-vec"));
  spatt::SpattPlusBuilder<vec_t> builder(&dl, program.get<std::string>("space"));
  LOG("Start building index");
  builder.BuildIndex({.max_N_      = dl.nb_,
                         .wp_      = program.get<size_t>("window-top"),
                         .M_       = program.get<size_t>("out-degree"),
                         .local_M_ = program.get<size_t>("local-m")},
      {.efc_             = program.get<size_t>("ef-construction"),
          .threads_      = program.get<int>("threads")});
  auto stat = builder.status_;
  LOG(fmt::format("Index building finished, time: {}, average distance computation: {}, average hop: {}, average out "
                  "degree: {}",
          stat.run_time_,
          stat.dist_computation_,
          stat.hop_,
          stat.M_));
  builder.SaveIndex(program.get<std::string>("output"));
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser program("buildindex");
  program.add_argument("-db", "--database").help("Database name, e.g.: sift1m, sift1b").required();
  program.add_argument("-vt", "--vec-type").help("Vector type, supported types: fvecs, bvecs, ivecs").required();
  program.add_argument("-at", "--attr-type").help("Attribute type, supported types: int, float, timestamp").required();
  program.add_argument("-bv", "--base-vec").help("Base vector path").required();
  program.add_argument("-s", "--space").help("Space type, supported spaces: l2, ip").required();
  program.add_argument("-o", "--output").help("Index output prefix").required();
  program.add_argument("-m", "--out-degree").help("Out degree for each node").default_value(16).scan<'u', size_t>();
  program.add_argument("-lm", "--local-m")
      .help("window boosting base")
      .default_value(1)
      .scan<'u', size_t>();
  program.add_argument("-efc", "--ef-construction").help("Construction ef").default_value(100).scan<'u', size_t>();
  program.add_argument("-wp", "--window-top")
      .help("number of top window graph")
      .default_value(1)
      .scan<'u', size_t>();
  program.add_argument("-t", "--threads").help("Number of threads").default_value(1).scan<'i', int>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    std::abort();
  }
  if (program.get<std::string>("vec-type") == "fvecs") {
    if (program.get<std::string>("attr-type") == "int") {
      RunBuilding<float>(program);
    } else {
      LOG("Unsupported attribute type");
      std::abort();
    }
  } else {
    LOG("Unsupported vector type");
    std::abort();
  }
}