#include <omp.h>
#include "hnswlib/hnswalg.h"
#include "common/dataloader.hh"
#include "common/type.hh"
#include "common/result.hh"
#include "apps/gtmanager.hh"
#include "argparse/argparse.hpp"

// def local_intrinsic_dimensionality(base_vec, query_vec, gt_range, k, dist_func):
//     query_num = 1000
//     ave_lid = 0
//     for i in range(query_num):
//         # print("Query", i)
//         query = query_vec[i]
//         gt = gt_range[i]
//         distances = []
//         for j in gt:
//             distances.append(dist_func(query, base_vec[j]))
//         distances = np.array(distances)
//         # print("Distances shape:", distances.shape)
//         distances = np.sort(distances)
//         assert (distances[-1] >= distances[0])
//         assert (distances[0] > 0)
//         sum_log_frac = 0
//         for j in range(k):
//             sum_log_frac += np.log((distances[j]) / distances[-1])
//         if sum_log_frac == 0:
//             lid = 0
//         else:
//             lid = -k / sum_log_frac
//         ave_lid += lid
//     ave_lid /= query_num
//     return ave_lid

#define IS_ZERO(x) ((x) < 1e-6 && (x) > -1e-6)

double distance_l2(const float *x, const float *y, size_t d)
{
  double dist = 0;
  for (size_t i = 0; i < d; i++) {
    dist += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return std::sqrt(dist);
}

double distance_ip(const float *x, const float *y, size_t d)
{
  double dist = 0;
  for (size_t i = 0; i < d; i++) {
    dist += x[i] * y[i];
  }
  return 1.0 - dist;
}

double distance_cos(const float *x, const float *y, size_t d)
{
  double dist   = 0;
  double norm_x = 0;
  double norm_y = 0;
  for (size_t i = 0; i < d; i++) {
    dist += x[i] * y[i];
    norm_x += x[i] * x[i];
    norm_y += y[i] * y[i];
  }
  return 1.0 - dist / (std::sqrt(norm_x) * std::sqrt(norm_y));
}

class DistFunc
{
public:
  DistFunc(std::string space, size_t d) : space_(space), d_(d) {}
  auto operator()(const float *x, const float *y) -> double
  {
    if (space_ == "l2") {
      return distance_l2(x, y, d_);
    } else if (space_ == "ip") {
      return distance_ip(x, y, d_);
    } else if (space_ == "cos") {
      return distance_cos(x, y, d_);
    } else {
      LOG("Unsupported space");
      std::abort();
    }
  }

private:
  std::string space_;
  size_t      d_;
};

auto LocalIntrinsicDimensionality(
    spatt::DataLoader<float> &dl, spatt::GTManager<float> &gt_manager, size_t k, DistFunc &distance) -> double
{
  size_t query_num = 1000;
  double ave_lid   = 0;

  for (size_t i = 0; i < query_num; i++) {
    auto               query = dl.GetQueryVecByID(i);
    auto               gt    = gt_manager.GetGT()[i];
    std::vector<float> distances;
    for (auto j : gt) {
      distances.push_back(distance(query, dl.GetBaseVecByID(j)));
    }
    std::sort(distances.begin(), distances.end());
    assert(distances.back() >= distances.front());
    assert(distances.front() > 0);
    double sum_log_frac = 0;
    for (size_t j = 0; j < k; j++) {
      if (distances[j] == 0) {
        continue;
      }
      if (distances.back() == 0) {
        sum_log_frac = 0;
        break;
      }
      double frac = distances[j] / distances.back();
      if (frac <= 0 || IS_ZERO(frac)) {
        continue;
      }
      sum_log_frac += std::log(distances[j] / distances.back());
    }
    double lid = 0;
    LOG(fmt::format("Query: {}, Sum log frac: {}", i, sum_log_frac));
    if (IS_ZERO(sum_log_frac)) {
      lid = 0.0;
    } else {
      ave_lid = ave_lid - k / sum_log_frac;
    }
    // LOG(fmt::format("Query: {}, LID: {}", i, lid));
  }
  ave_lid /= query_num;
  return ave_lid;
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
  program.add_argument("-k", "--k").help("Top k").default_value(10).scan<'u', size_t>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    std::abort();
  }

  spatt::DataLoader<float> dl(program.get<std::string>("database"),
      program.get<std::string>("vec-type"),
      program.get<std::string>("attr-type"));
  dl.Loadfves(program.get<std::string>("base-vec"));
  dl.Loadfves(program.get<std::string>("query-vec"), true);
  std::vector<double> lids;
  for (int frac = 0; frac <= 17; frac++) {
    std::string query_attr = program.get<std::string>("query-attr") + std::to_string(frac) + ".bin";
    dl.LoadQueryFilter(query_attr);
    spatt::GTManager<float> gt_manager(&dl,
        fmt::format("{}{}.bin", program.get<std::string>("ground-truth"), frac),
        program.get<size_t>("k"),
        program.get<std::string>("space"));
    gt_manager.LoadGT();
    DistFunc distance(program.get<std::string>("space"), dl.d_);
    double   lid = LocalIntrinsicDimensionality(dl, gt_manager, program.get<size_t>("k"), distance);
    lids.push_back(lid);
  }
  for (size_t i = 0; i < lids.size(); i++) {
    LOG(fmt::format("Frac: {}, LID: {}", i, lids[i]));
  }
}
