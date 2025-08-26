#pragma once

#include <algorithm>
#include <cmath>
#include <omp.h>
#include "index.hh"

namespace spatt {

template <typename att_t>
using RangeLevelVec =
    std::vector<std::pair<std::pair<std::unique_ptr<QueryFilter<att_t>>, std::unique_ptr<QueryFilter<att_t>>>, int>>;

template <typename vec_t = float>
class SpattIndexBuilder : public SpattIndex<vec_t>
{
public:
  SpattIndexBuilder(DataLoader<vec_t> *dl, const std::string &space) : SpattIndex<vec_t>(dl, space) {}

  void BuildIndex(const IndexParameters &iparam, const BuildParametersRuntime &brparam);

  void AddPoint(label_t label, const vec_t *v, bool replace_deleted = false);

private:
  /**
   * @brief prune the candidates in current subgraph by heuristic
   *
   * @param candidates
   * @param M
   * @return std::vector<dist_id_pair>
   */
  auto PruneByHeuristic(std::vector<dist_id_pair> &candidates, const size_t M) -> std::vector<dist_id_pair>;

public:
  BuildParametersRuntime brparam_;
  RuntimeStatus          status_;
};

template <typename vec_t>
void SpattIndexBuilder<vec_t>::BuildIndex(const IndexParameters &iparam, const BuildParametersRuntime &brparam)
{
  this->iparam_ = iparam;
  LOG(fmt::format("max_N_: {}, wp: {}, M_: {}, local_M_: {}",
      this->iparam_.max_N_,
      this->iparam_.wp_,
      this->iparam_.M_,
      this->iparam_.local_M_));
  this->brparam_ = brparam;
  LOG(fmt::format("efc_: {}, threads_: {}",
      this->brparam_.efc_,
      this->brparam_.threads_));
  this->elemperlinklist_  = (this->iparam_.M_ + 1) * (this->iparam_.wp_ + 1);
  this->sizelinklistsmem_ = this->iparam_.max_N_ * this->elemperlinklist_ * sizeof(label_t);
  ASSERT_MSG(this->sizelinklistsmem_ < std::numeric_limits<size_t>::max(), std::to_string(this->sizelinklistsmem_));
  ASSERT_MSG(this->linklistsmemory_ == nullptr, "linklistsmemory_ should be nullptr");
  LOG(fmt::format("elemperlinklist_: {}, sizelinklistsmem_: {}", this->elemperlinklist_, this->sizelinklistsmem_));
  // this->linklistsmemory_ = (label_t *)malloc(this->sizelinklistsmem_);
  this->linklistsmemory_ = (label_t *)glass::alloc2M(this->sizelinklistsmem_);
  if (this->linklistsmemory_ == nullptr) {
    LOG("Failed to allocate memory for linklistsmemory_");
    std::abort();
  }
  const auto storage = this->storage_;
  if (storage->nb_ > this->iparam_.max_N_) {
    LOG(fmt::format("nb_ > max_N_, nb_: {}, max_N_: {}", storage->nb_, this->iparam_.max_N_));
    std::abort();
  }

  // build index
  this->linklist_locks_ = std::vector<std::mutex>(this->iparam_.max_N_);
  this->visited_pool_.Init(this->iparam_.max_N_);
  auto start = std::chrono::high_resolution_clock::now();

  std::atomic<int> processed{0};
#pragma omp parallel for num_threads(brparam_.threads_) schedule(dynamic) shared(processed, storage)
  for (int i = 0; i < storage->nb_; ++i) {
    this->AddPoint(i, storage->GetBaseVecByID(i));
    processed++;
    if (processed % 1000 == 0) {
      LOG(fmt::format("Processed: {}/{}", processed.load(), storage->nb_));
    }
  }

  status_.run_time_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  // calculate average out degree for each layer
  for (int layer = 0; layer <= this->iparam_.wp_; ++layer) {
    int M = 0;
    for (size_t i = 0; i < storage->nb_; ++i) {
      auto ll = &this->linklistsmemory_[i * this->elemperlinklist_ + layer * (this->iparam_.M_ + 1)];
      M += ll[this->iparam_.M_];
    }
    LOG(fmt::format("Layer: {}, average M: {}", layer, M / storage->nb_));
    status_.M_ += M;
  }
  status_.dist_computation_ /= storage->nb_;
  status_.hop_ /= storage->nb_;
  status_.M_ /= storage->nb_;
}

template <typename vec_t>
void SpattIndexBuilder<vec_t>::AddPoint(label_t label, const vec_t *v, bool replace_deleted)
{
  LOG_DBG(fmt::format("AddPoint: {}", label));
  // add point to each layer
  auto storage = this->storage_;
  auto cur_num = this->atomic_curvec_num_.fetch_add(1);
  if (cur_num == 0) {
    std::unique_lock<std::mutex> lock(this->linklist_locks_[label]);
    for (int layer = 0; layer <= this->iparam_.wp_; ++layer) {
      auto ll              = &this->linklistsmemory_[label * this->elemperlinklist_ + layer * (this->iparam_.M_ + 1)];
      ll[this->iparam_.M_] = 0;
    }
    return;
  }
  unsigned                   seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  for (int layer = this->iparam_.wp_; layer >= 0; --layer) {
    auto M = this->iparam_.M_;
    // calculate window
    int                                half_window_size = std::pow(this->iparam_.local_M_, layer);
    int                                s_pos            = label > half_window_size ? label - half_window_size : 0;
    int                                e_pos            = label > 1 ? label - 1 : 0;
    std::uniform_int_distribution<int> distribution(s_pos, e_pos);
    std::vector<label_t>               entry_points;
    if (entry_points.empty()) {
      for (int i = 0; i < std::min(e_pos - s_pos + 1, 3); ++i) {
        entry_points.emplace_back(distribution(generator));
      }
    }

    auto allc = this->template SearchCandidates<true>(
        entry_points, v, {static_cast<unsigned int>(s_pos), static_cast<unsigned int>(e_pos)}, {layer, static_cast<int>(this->iparam_.wp_)}, this->brparam_.efc_, status_, true);

    LOG_DBG(fmt::format("label: {}, layer: {}, allc: {}", label, layer, DistIDVecToString(allc)));
    // prune candidates by heuristic
    auto pruned = PruneByHeuristic(allc, M / 2);
    LOG_DBG(fmt::format("label: {}, layer: {}, pruned: {}", label, layer, DistIDVecToString(pruned)));
    ASSERT_MSG(pruned.size() <= M, std::to_string(pruned.size()));

    {  // add neighbors for the current layer
      std::unique_lock<std::mutex> lock(this->linklist_locks_[label]);
      label_t                     *ll = this->linklistsmemory_ + label * this->elemperlinklist_ + layer * (M + 1);
      ll[M]                           = (label_t)pruned.size();
      for (int i = 0; i < pruned.size(); ++i) {
        if (pruned[i].id_ == label) {
          throw std::runtime_error("pruned[i].id_ == label");
        }
        ll[i] = pruned[i].id_;
      }
    }

    // // add and prune for neighbors in the same layer
    for (const auto &[nn_d, nn_i] : pruned) {
      std::lock_guard<std::mutex> lock(this->linklist_locks_[nn_i]);
      auto                        nn_ll    = &this->linklistsmemory_[nn_i * this->elemperlinklist_ + layer * (M + 1)];
      auto                        nn_ll_sz = nn_ll[M];
      if (nn_ll_sz < M) {
        nn_ll[nn_ll_sz] = label;
        nn_ll[M]++;
      } else {
        std::vector<dist_id_pair> nn_allc;
        nn_allc.reserve(nn_ll_sz + 1);
        for (int i = 0; i < nn_ll_sz; ++i) {
          nn_allc.emplace_back(
              this->CalculateDist(storage->GetBaseVecByID(nn_i), storage->GetBaseVecByID(nn_ll[i])), nn_ll[i]);
        }
        nn_allc.emplace_back(nn_d, label);
        auto nn_pruned = PruneByHeuristic(nn_allc, M);
        ASSERT_MSG(nn_pruned.size() <= M, std::to_string(nn_pruned.size()));
        for (int i = 0; i < nn_pruned.size(); ++i) {
          nn_ll[i] = nn_pruned[i].id_;
        }
        nn_ll[M] = nn_pruned.size();
      }
    }
  }
}

template <typename vec_t>
inline auto SpattIndexBuilder<vec_t>::PruneByHeuristic(
    std::vector<dist_id_pair> &candidates, const size_t M) -> std::vector<dist_id_pair>
{
  // dist is the distance from query q to a, id is the id of a
  // prune metric: scan the candidates sequentially, currently visited node xa, if there is a node xb in the
  // result set requires: dist(q, xa) < dist(q, xb) and dist(xa, xb) < dist(q, xa), then it should be pruned,
  // as candidates are ordered by  dist(q, a), we can prune the candidates by just checking the latter condition
  if (candidates.size() <= M) {
    return candidates;
  }
  if (M == 0) {
    return {};
  }
  if (M == 1) {
    return {candidates[0]};
  }
  // ensure the candidates are sorted by distance
  std::sort(candidates.begin(), candidates.end());
  std::vector<dist_id_pair> pruned;
  for (const auto &[db, ib] : candidates) {
    if (pruned.size() >= M) {
      break;
    }
    bool good = true;
    for (const auto &[da, ia] : pruned) {
      auto curdist = this->CalculateDist(this->storage_->GetBaseVecByID(ib), this->storage_->GetBaseVecByID(ia));
      status_.dist_computation_++;
      if (curdist < db) {  // i == ib is to avoid repeated points
        good = false;
        break;
      }
    }
    if (good) {
      pruned.emplace_back(db, ib);
    }
  }
  return pruned;
}
}  // namespace spatt
