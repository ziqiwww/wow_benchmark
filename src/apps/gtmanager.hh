
#pragma once

#include <omp.h>
#include "common/dataloader.hh"
#include "common/micro.hh"
#include "hnswlib/hnswalg.h"

namespace spatt {
template <typename vec_t = float>
class GTManager
{
public:
  explicit GTManager(DataLoader<vec_t> *dl, std::string gt_file, size_t k, const std::string space)
      : dl_(dl), gt_file_(std::move(gt_file)), k_(k)
  {
    if (space == "l2") {
      space_ = new hnswlib::L2Space(dl_->d_);
    } else if (space == "ip") {
      space_ = new hnswlib::InnerProductSpace(dl_->d_);
    } else if (space == "cos") {
      space_ = new hnswlib::InnerProductSpace(dl_->d_);
    } else {
      LOG(fmt::format("Unsupported space: {}", space));
      std::abort();
    }
  }

  ~GTManager() { delete space_; }

  DISABLE_COPY_MOVE_AND_ASSIGN(GTManager);

  void LoadGT();

  void GenerateGT(bool is_ordered);

  void SaveGT();

  inline auto GetGTByQueryID(size_t id) -> const std::vector<label_t> & { return gt_[id]; }

  [[nodiscard]] inline auto GetGT() const -> const std::vector<std::vector<label_t>> & { return gt_; }

private:
  DataLoader<vec_t>         *dl_;
  std::string                       gt_file_;
  size_t                            k_;
  std::vector<std::vector<label_t>> gt_{};
  hnswlib::SpaceInterface<vec_t>   *space_{};
};

template <typename vec_t>
inline void GTManager<vec_t>::LoadGT()
{
  std::ifstream in(gt_file_, std::ios::binary);
  if (!in.is_open()) {
    LOG(fmt::format("Failed to open {}", gt_file_));
    std::abort();
  }
  while (!in.eof()) {
    int n;
    in.read(reinterpret_cast<char *>(&n), sizeof(int));
    if (in.eof()) {
      break;
    }
    std::vector<label_t> gt;
    gt.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      int id;
      in.read(reinterpret_cast<char *>(&id), sizeof(int));
      gt.emplace_back(static_cast<label_t>(id));
    }
    gt_.emplace_back(gt);
  }
  LOG(fmt::format("Loaded {} ground truth", gt_.size()));
  std::string eg;
  for (const auto &id : gt_[0]) {
    eg += fmt::format("{}, ", id);
  }
  LOG(fmt::format("Example ground truth, size {}: {}", gt_[0].size(), eg));
  in.close();
}

template <typename vec_t>
inline void GTManager<vec_t>::GenerateGT(bool is_ordered)
{
  // generate ground truth for each query according to distance to query vec and query filter
  auto             start = std::chrono::high_resolution_clock::now();
  std::atomic<int> counter{0};
  gt_.resize(dl_->nq_);
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic) shared(gt_, dl_, space_, k_, counter)
  for (size_t i = 0; i < dl_->nq_; ++i) {
    auto                      query = dl_->GetQueryVecByID(i);
    auto                      rng   = dl_->GetQueryFilterByID(i);
    std::vector<dist_id_pair> result;
    for (int j = 0; j < dl_->nb_; ++j) {
      if ((j < rng.l_ || j > rng.u_)) {
        continue;
      }
      auto dist = space_->get_dist_func()(query, dl_->GetBaseVecByID(j), space_->get_dist_func_param());
      PUSH_HEAP(result, dist, j);
      if (result.size() > k_) {
        POP_HEAP(result);
      }
    }
    if (result.size() > k_) {
      LOG(fmt::format("Result size {} > k_ {}, abort", result.size(), k_));
      std::abort();
    }
    std::vector<label_t> gt;
    for (size_t j = 0; j < std::min(k_, result.size()); ++j) {
      gt.emplace_back(result[j].id_);
    }
    gt_[i] = std::move(gt);
    if (++counter % 100 == 0) {
      LOG(fmt::format("Groundtruth: {}/{}", counter.load(), dl_->nq_));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  ASSERT_MSG(gt_.size() == dl_->nq_, fmt::format("Generated {} ground truth, expected {}", gt_.size(), dl_->nq_));
  LOG(fmt::format(
      "Generated ground truth for {} queries in {}s", dl_->nq_, std::chrono::duration<double>(end - start).count()));
  std::string eg;
  for (const auto &id : gt_[0]) {
    eg += fmt::format("{}, ", id);
  }
  LOG(fmt::format("Example ground truth, size {}: {}", gt_[0].size(), eg));
}

template <typename vec_t>
inline void GTManager<vec_t>::SaveGT()
{
  std::ofstream out(gt_file_, std::ios::binary);
  if (!out.is_open()) {
    LOG(fmt::format("Failed to open {}", gt_file_));
    std::abort();
  }
  for (const auto &gt : gt_) {
    int n = static_cast<int>(gt.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(int));
    for (const auto &id : gt) {
      int id_ = static_cast<int>(id);
      out.write(reinterpret_cast<const char *>(&id_), sizeof(int));
    }
  }
  out.close();
  LOG(fmt::format("Saved {} ground truth", gt_.size()));
}
}  // namespace spatt