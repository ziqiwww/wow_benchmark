#include "gtmanager.hh"
#include "argparse/argparse.hpp"

template <typename vec_t, typename att_t>
void RunGTManager(const argparse::ArgumentParser &program)
{
  // check vec format and create dataloader
  spatt::DataLoader<vec_t> dl(program.get<std::string>("database"),
      program.get<std::string>("vec-type"),
      program.get<std::string>("attr-type"));
  dl.Loadfves(program.get<std::string>("base-vec"));
  dl.Loadfves(program.get<std::string>("query-vec"), true);
  dl.LoadQueryFilter(program.get<std::string>("query-attr"));
  spatt::GTManager<vec_t> gt_manager(
      &dl, program.get<std::string>("ground-truth"), program.get<size_t>("k"), program.get<std::string>("space"));
  // generate ground truth
  bool is_ordered = program.get<int>("is-ordered") == 1;
  gt_manager.GenerateGT(is_ordered);
  // save ground truth
  gt_manager.SaveGT();
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
  program.add_argument("-gt", "--ground-truth").help("Ground truth path").required();
  program.add_argument("-s", "--space").help("Space type, supported spaces: l2, ip").required();
  program.add_argument("-io", "--is-ordered").help("Is meta data ordered").default_value(1).scan<'i', int>();
  program.add_argument("-k", "--k").help("Top k").default_value(10).scan<'u', size_t>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    std::abort();
  }

  if (program.get<std::string>("vec-type") == "fvecs") {
    if (program.get<std::string>("attr-type") == "int") {
      RunGTManager<float, int>(program);
    } else if (program.get<std::string>("attr-type") == "float") {
      RunGTManager<float, float>(program);
    } else {
      LOG("Unsupported attribute type");
      std::abort();
    }
    // it seems that hnswlib only supports float type
    //   } else if (program.get<std::string>("vec-type") == "bvecs") {
    //     if (program.get<std::string>("attr-type") == "int") {
    //       RunGTManager<uint8_t, int>(program);
    //     } else if (program.get<std::string>("attr-type") == "float") {
    //       RunGTManager<uint8_t, float>(program);
    //     } else {
    //       LOG("Unsupported attribute type");
    //       std::abort();
    //     }
    //   } else {
    //     LOG("Unsupported vector type");
    //     std::abort();
  }

  return 0;
}