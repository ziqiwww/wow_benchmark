#pragma once

#include <algorithm>
#include <cmath>
#include <omp.h>
#include "spattplusindex.hh"

namespace spatt {

template <typename vec_t = float>
class SpattPlusBuilder : public SpattPlus<vec_t>
{
public:
  SpattPlusBuilder(DataLoader<vec_t> *dl, const std::string &space) : SpattPlus<vec_t>(dl, space) {}

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
void SpattPlusBuilder<vec_t>::BuildIndex(const IndexParameters &iparam, const BuildParametersRuntime &brparam)
{
  this->iparam_ = iparam;
  LOG(fmt::format("max_N_: {}, wp: {}, M_: {}, local_M_: {}",
      this->iparam_.max_N_,
      this->iparam_.wp_,
      this->iparam_.M_,
      this->iparam_.local_M_));
  this->brparam_ = brparam;
  LOG(fmt::format("efc_: {}, threads_: {}", this->brparam_.efc_, this->brparam_.threads_));
  this->elemperlinklist_  = (this->iparam_.M_ + 1) * (this->iparam_.wp_ + 1);
  this->sizelinklistsmem_ = this->iparam_.max_N_ * this->elemperlinklist_ * sizeof(label_t);
  ASSERT_MSG(this->sizelinklistsmem_ < std::numeric_limits<size_t>::max(), std::to_string(this->sizelinklistsmem_));
  ASSERT_MSG(this->linklistsmemory_ == nullptr, "linklistsmemory_ should be nullptr");
  LOG(fmt::format("elemperlinklist_: {}, sizelinklistsmem_: {}", this->elemperlinklist_, this->sizelinklistsmem_));
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

  // initiliaze window size
  this->window_size_.resize(iparam.wp_ + 1);
  this->window_size_[0] = 2;
  for (int i = 1; i <= iparam.wp_; ++i) {
    this->window_size_[i] = iparam.local_M_ * this->window_size_[i - 1];
  }
  
  this->linklist_locks_ = std::vector<std::mutex>(this->iparam_.max_N_);
  this->visited_pool_.Init(this->iparam_.max_N_);
  // this->order_table_ = new VectorOrderTable<att_t>(storage->ab_);
  this->order_table_ = new WBTreeOrderTable(this->iparam_.max_N_);

  // build index
  // shuffle ids to simulate random insert
  std::vector<label_t> shuffled_idxs(storage->nb_);
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0);
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), std::mt19937(std::random_device()()));
  auto             start = std::chrono::high_resolution_clock::now();
  std::atomic<int> processed{0};
  LOG("Start building index");
#pragma omp parallel for num_threads(brparam_.threads_) schedule(dynamic) shared(processed, storage, shuffled_idxs)
  for (label_t i = 0; i < storage->nb_; ++i) {
    auto idx = shuffled_idxs[i];
    AddPoint(idx, storage->GetBaseVecByID(idx));
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
void SpattPlusBuilder<vec_t>::AddPoint(label_t label, const vec_t *v, bool replace_deleted)
{
  // add point to each layer
  // LOG(fmt::format("AddPoint: {}", label));
  auto storage        = this->storage_;
  int  max_level_copy = -1;
  {
    std::unique_lock<std::mutex> lock(this->max_layer_lock_);
    auto                         cur_num = this->curvec_num_++;
    if (cur_num == 0) {
      std::lock_guard<std::mutex> lock(this->linklist_locks_[label]);
      for (int layer = 0; layer <= this->iparam_.wp_; ++layer) {
        auto ll              = &this->linklistsmemory_[label * this->elemperlinklist_ + layer * (this->iparam_.M_ + 1)];
        ll[this->iparam_.M_] = 0;
      }
      this->order_table_->InsertLabel(label);
      return;
    }
    if (cur_num > this->window_size_[this->cur_max_layer_]) {
      LOG(fmt::format(
          "layer copy trigered: cur_num: {}, 2 * local_M^wp: {}", cur_num, this->window_size_[this->cur_max_layer_]));
      // triger the layer copy process
      if (this->cur_max_layer_ == this->iparam_.wp_) {
        throw std::runtime_error("no enough space for new layer");
      }
      this->cur_max_layer_++;
      // copy this->cur_max_layer - 1 to this->cur_max_layer_ for points in order table
      {
        // std::unique_lock<std::mutex> lock(this->order_table_->lock_);
        for (int lower_label = 0; lower_label < this->iparam_.max_N_; ++lower_label) {
          auto lower_link_list = this->linklistsmemory_ + lower_label * this->elemperlinklist_ +
                                 (this->cur_max_layer_ - 1) * (this->iparam_.M_ + 1);
          if (lower_link_list[this->iparam_.M_] == 0) {
            continue;
          }
          auto upper_link_list = this->linklistsmemory_ + lower_label * this->elemperlinklist_ +
                                 this->cur_max_layer_ * (this->iparam_.M_ + 1);
          memcpy(upper_link_list, lower_link_list, (this->iparam_.M_ + 1) * sizeof(label_t));
        }
      }
    }
    max_level_copy = this->cur_max_layer_;
  }
  if (max_level_copy == -1) {
    throw std::runtime_error("max_level_copy == -1");
  }
  std::vector<std::vector<dist_id_pair>> tmp_linklist(max_level_copy + 1);
  std::vector<dist_id_pair> prev_layer_allc;
  {
    for (int layer = max_level_copy; layer >= 0; --layer) {
      auto M = this->iparam_.M_;
      // calculate window
      int                  half_window_size = std::pow(this->iparam_.local_M_, layer);
      std::vector<label_t> entry_points;
      auto query_rng = this->order_table_->GetWindowedFilterAndEntries(label, half_window_size, entry_points);
      /**
       * @brief building optimization
       * we can simply use the following code to get the nearest candidates for all layers:
       * 
       * auto allc = this->SearchCandidatesKNN(v, label, {s_pos, e_pos}, this->brparam_.efc_, status_, true);
       * 
       * but the indexing time is log^2(n). The following code ensures in the worst case, the time is log^2 (n).
       * It first check the previously retrieved candidates, if the number of candidates is larger than M, we can
       * directly use them, otherwise, we need to search on the incomplete graph.
       * 
       * another optimization opportunity resides in the setting of the entry points, we can use the filtered candidates as entry points,
       * however, the acceleration is not significant and the current implementation is simple and efficient enough.
       */
      std::vector<dist_id_pair> tmp_prev_layer_allc;
      Bitset prev_allc_record(this->iparam_.max_N_);
      prev_allc_record.ClearRange(query_rng.l_, query_rng.u_);
      for (const auto &[d, i] : prev_layer_allc) {
        if (i >= query_rng.l_ && i <= query_rng.u_) {
          tmp_prev_layer_allc.emplace_back(d, i);
          prev_allc_record.Set(i);
        }
      }
      prev_layer_allc = std::move(tmp_prev_layer_allc);
      std::vector<dist_id_pair> allc;
      if(prev_layer_allc.size() > M) {
        allc = prev_layer_allc;
      }
      else{
        allc = this->template SearchCandidates<true>(
                  entry_points, v, query_rng, {layer, max_level_copy}, this->brparam_.efc_, status_, label);
        // add new point in allc to prev_layer_allc
        for (const auto &[d, i] : allc) {
          if (i == label) {
            throw std::runtime_error("i == label");
          }
          if (!prev_allc_record.Test(i)) {
            prev_layer_allc.emplace_back(d, i);
          }
        }
        allc = prev_layer_allc;
      }
      LOG_DBG(fmt::format("label: {}, layer: {}, allc: {}", label, layer, DistIDVecToString(allc)));
      // prune candidates by heuristic
      auto pruned = PruneByHeuristic(allc, M / 2);
      LOG_DBG(fmt::format("label: {}, layer: {}, pruned: {}", label, layer, DistIDVecToString(pruned)));
      ASSERT_MSG(pruned.size() <= M, std::to_string(pruned.size()));
      tmp_linklist[layer] = std::move(pruned);
    }
  }

  {
    auto                         M = this->iparam_.M_;
    std::unique_lock<std::mutex> lock(this->linklist_locks_[label]);
    for (int layer = 0; layer <= max_level_copy; ++layer) {
      {  // add neighbors for the current layer
        label_t *ll = this->linklistsmemory_ + label * this->elemperlinklist_ + layer * (M + 1);
        ll[M]       = (label_t)tmp_linklist[layer].size();
        for (int i = 0; i < ll[M]; ++i) {
          if (tmp_linklist[layer][i].id_ == label) {
            throw std::runtime_error("pruned[i].id_ == label");
          }
          if (ll[i]) {
            LOG_DBG(fmt::format("current label: {}, layer: {}, ll[M]: {}, pruned: {}",
                label,
                layer,
                ll[M],
                DistIDVecToString(tmp_linklist[layer])));
            throw std::runtime_error("newly added point should have blank link list");
          }
          ll[i] = tmp_linklist[layer][i].id_;
        }
      }

      // // add and prune for neighbors in the same layer
      for (const auto &[nn_d, nn_i] : tmp_linklist[layer]) {
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
          int                  half_window_size = std::pow(this->iparam_.local_M_, layer);
          nn_allc        = this->order_table_->GetInWindowCandidates(nn_allc, nn_i, half_window_size);
          nn_allc.emplace_back(nn_d, label);
          auto nn_pruned = PruneByHeuristic(nn_allc, M);
          ASSERT_MSG(nn_pruned.size() <= M, std::to_string(nn_pruned.size()));
          for (int i = 0; i < nn_pruned.size(); ++i) {
            nn_ll[i] = nn_pruned[i].id_;
          }
          nn_ll[M] = (label_t)nn_pruned.size();
        }
      }
    }
  }
  this->order_table_->InsertLabel(label);
}

template <typename vec_t>
inline auto SpattPlusBuilder<vec_t>::PruneByHeuristic(
    std::vector<dist_id_pair> &candidates, const size_t M) -> std::vector<dist_id_pair>
{
  /**
   * @brief 
   *   dist is the distance from query q to a, id is the id of a
   *   prune metric: scan the candidates sequentially, currently visited node xa, if there is a node xb in the
   *   result set requires: dist(q, xa) < dist(q, xb) and dist(xa, xb) < dist(q, xa), then it should be pruned,
   *   as candidates are ordered by  dist(q, a), we can prune the candidates by just checking the latter condition
   */

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
  pruned.reserve(M);
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
