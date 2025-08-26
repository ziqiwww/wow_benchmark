#pragma once

#include <mutex>
#include <unordered_set>
#include "hnswlib/hnswalg.h"
#include "common/dataloader.hh"
#include "common/disk.hh"
#include "common/type.hh"
#include "common/config.hh"
#include "common/memory.hh"
#include "apps/gtmanager.hh"
#include "order_table.hh"

namespace spatt {

/// @brief record run time, distance computation, hop
struct RuntimeStatus
{
  double run_time_{0.0};
  size_t dist_computation_{0};
  size_t hop_{0};
  double M_{0.0};
};

struct IndexParameters
{
  // max number of elements in the index
  size_t max_N_;
  // top layer
  size_t wp_;
  // out degree
  size_t M_;
  // window boosting base
  size_t local_M_;
};

template <typename vec_t = float>
class SpattPlus
{

public:
  SpattPlus(DataLoader<vec_t> *dl, const std::string &space);

  ~SpattPlus();

  inline auto CalculateDist(const vec_t *pVect1, const vec_t *pVect2) -> dist_t;

  void SaveIndex(const std::string &prefix);

  void LoadIndex(const std::string &location);

protected:
  /**
   * @brief search for knn candidates in the current subgraph with meta and return the distance and id pairs
   *
   * @param v
   * @param meta
   * @param sparam
   * @return std::vector<dist_id_pair>
   */
  template <bool is_build>
  auto SearchCandidates(std::vector<label_t> &eps, const vec_t *v, const QueryFilter<label_t> &filter,
      const QueryFilter<int> &layer_rng, const size_t ef, RuntimeStatus &status,
      label_t ignore = -1) -> std::vector<dist_id_pair>;

private:

  auto IndexFileName() -> std::string;

public:
  IndexParameters                 iparam_;
  int                             cur_max_layer_{0};
  size_t                          curvec_num_{0};
  DataLoader<vec_t>       *storage_;
  hnswlib::SpaceInterface<vec_t> *space_;
  hnswlib::DISTFUNC<vec_t>        fstdistfunc_;
  void                           *dist_func_param_;
  size_t                          elemperlinklist_;  // should be equal to M + 1 without * sizeof(label_t)
  size_t                          sizelinklistsmem_;
  // last element is used to store the number of elements in the link list
  // we use label_t insread of dist_id_pair to save memory, dist calculation should be done on the fly during building
  label_t           *linklistsmemory_;
  OrderTable *order_table_;

  std::vector<std::mutex>  linklist_locks_;  // idx is label_t
  std::mutex               max_layer_lock_;
  VisitedPool<Bitset> visited_pool_;

  std::vector<size_t> window_size_;

#ifdef DEBUG
  // used for debugging
  GTManager<vec_t, att_t> *gt_manager_{nullptr};
  size_t                   cur_query_id_{0};
#endif
};

template <typename vec_t>
SpattPlus<vec_t>::SpattPlus(DataLoader<vec_t> *dl, const std::string &space) : storage_(dl)
{
  if (space == "l2") {
    space_ = new hnswlib::L2Space(storage_->d_);
  } else if (space == "ip") {
    space_ = new hnswlib::InnerProductSpace(storage_->d_);
  } else if (space == "cos") {
    space_ = new hnswlib::InnerProductSpace(storage_->d_);
  } else {
    LOG(fmt::format("Unsupported space: {}", space));
    std::abort();
  }
  fstdistfunc_     = space_->get_dist_func();
  dist_func_param_ = space_->get_dist_func_param();
  linklistsmemory_ = nullptr;
}

template <typename vec_t>
inline SpattPlus<vec_t>::~SpattPlus()
{
  free(linklistsmemory_);
  linklistsmemory_ = nullptr;
  delete space_;
  delete order_table_;
}

template <typename vec_t>
inline auto SpattPlus<vec_t>::CalculateDist(const vec_t *pVect1, const vec_t *pVect2) -> dist_t
{
  return fstdistfunc_(pVect1, pVect2, dist_func_param_);
}

template <typename vec_t>
inline void SpattPlus<vec_t>::SaveIndex(const std::string &prefix)
{
  std::string   index_file = fmt::format("{}/{}", prefix, IndexFileName());
  std::ofstream ofs(index_file, std::ios::binary);
  if (!ofs.is_open()) {
    LOG(fmt::format("Failed to open file: {}", index_file));
    std::abort();
  }
  WriteBinaryPOD(ofs, iparam_);
  WriteBinaryPOD(ofs, cur_max_layer_);
  WriteBinaryPOD(ofs, curvec_num_);
  WriteBinaryPOD(ofs, elemperlinklist_);
  WriteBinaryPOD(ofs, sizelinklistsmem_);
  for (size_t i = 0; i < iparam_.max_N_; ++i) {
    for (size_t j = 0; j < elemperlinklist_; ++j) {
      WriteBinaryPOD(ofs, linklistsmemory_[i * elemperlinklist_ + j]);
    }
  }
  order_table_->Serialize(ofs);
  ofs.close();
  LOG(fmt::format("Index saved to: {}", index_file));
}

template <typename vec_t>
inline void SpattPlus<vec_t>::LoadIndex(const std::string &location)
{
  std::ifstream ifs(location, std::ios::binary);
  if (!ifs.is_open()) {
    LOG(fmt::format("Failed to open file: {}", location));
    std::abort();
  }
  ReadBinaryPOD(ifs, iparam_);
  ReadBinaryPOD(ifs, cur_max_layer_);
  ReadBinaryPOD(ifs, curvec_num_);
  ReadBinaryPOD(ifs, elemperlinklist_);
  ReadBinaryPOD(ifs, sizelinklistsmem_);
  linklistsmemory_ = (label_t *)glass::alloc2M(sizelinklistsmem_);
  if (linklistsmemory_ == nullptr) {
    LOG("Failed to allocate memory for linklistsmemory_");
    std::abort();
  }
  for (size_t i = 0; i < iparam_.max_N_; ++i) {
    for (size_t j = 0; j < elemperlinklist_; ++j) {
      ReadBinaryPOD(ifs, linklistsmemory_[i * elemperlinklist_ + j]);
    }
  }
  order_table_    = new WBTreeOrderTable(iparam_.max_N_);
  order_table_->Deserialize(ifs);
  ifs.close();
  linklist_locks_ = std::vector<std::mutex>(iparam_.max_N_);
  visited_pool_.Init(iparam_.max_N_);
  visited_pool_.Return(visited_pool_.Get());
  LOG(fmt::format("Index loaded from: {}", location));
  LOG(fmt::format("Index parameters: max_N: {}, wp:{}, M: {}, local_M: {}",
      iparam_.max_N_,
      iparam_.wp_,
      iparam_.M_,
      iparam_.local_M_));
  LOG(fmt::format("M + 1: {}, linklist memory size: {}", elemperlinklist_, sizelinklistsmem_));
  // calculate average out degree for each layer
  for (int layer = 0; layer <= iparam_.wp_; ++layer) {
    size_t M = 0;
    for (size_t i = 0; i < storage_->nb_; ++i) {
      auto ll = &linklistsmemory_[i * elemperlinklist_ + layer * (iparam_.M_ + 1)];
      M += ll[iparam_.M_];
    }
    LOG(fmt::format("Layer: {}, average M: {}", layer, M / storage_->nb_));
  }

  // initiliaze window size
  window_size_.resize(iparam_.wp_ + 1);
  window_size_[0] = 2;
  for (int i = 1; i <= iparam_.wp_; ++i) {
    window_size_[i] = iparam_.local_M_ * window_size_[i - 1];
  }
}

template <typename vec_t>
inline auto SpattPlus<vec_t>::IndexFileName() -> std::string
{
  // generate index file name based on the search parameter and template parameters
  return fmt::format("pp_{}_{}_{}_{}_{}_{}_{}_{}.index",
      storage_->db_name_,
      storage_->v_type_,
      storage_->a_type_,
      storage_->d_,
      storage_->nb_,
      iparam_.wp_,
      iparam_.M_,
      iparam_.local_M_);
}

template <typename vec_t>
template <bool is_build>
inline auto SpattPlus<vec_t>::SearchCandidates(std::vector<label_t> &eps, const vec_t *v,
    const QueryFilter<label_t> &filter, const QueryFilter<int> &layer_rng, const size_t ef, RuntimeStatus &status,
    label_t ignore) -> std::vector<dist_id_pair>
{
  ASSERT_MSG(layer_rng.l_ >= 0 && layer_rng.u_ <= iparam_.wp_ && layer_rng.l_ <= layer_rng.u_, "Invalid layer range");
  auto visited = visited_pool_.Get();

  // in reality, range is not id in out-of-order case, for the vectors are not sorted by id. but it's ok for evaluation
  if (is_build) {
    visited->Clear();
  } else {
    visited->ClearRange(filter.l_, filter.u_);
  }

  if (is_build && ignore != -1) {
    visited->Set(ignore);
  }
  // candidates is a minheap with negative distance, dist = -distof(a, query)
  std::vector<dist_id_pair> candidates;
  // result is a max heap with positive distance, dist = distof(a, query)
  std::vector<dist_id_pair> result;
  for (auto ep : eps) {
    auto d = fstdistfunc_(v, storage_->GetBaseVecByID(ep), dist_func_param_);
    status.dist_computation_++;
    PUSH_HEAP(candidates, -d, ep);
    PUSH_HEAP(result, d, ep);
    visited->Set(ep);
  }

  auto res_max_dist = TOP_HEAP(result).dist_;
  while (!candidates.empty()) {
    auto [dist, id] = TOP_HEAP(candidates);
    ASSERT_MSG((-dist) >= 0, std::to_string(dist));
    ASSERT_MSG(visited->Test(id), std::to_string(id));
    bool should_stop = is_build ? ((-dist) > res_max_dist && result.size() == ef) : ((-dist) > res_max_dist);
    if (should_stop) {
      break;
    }
    POP_HEAP(candidates);
    status.hop_++;

    if (is_build)
      linklist_locks_[id].lock();
    int neighbor_cnt = 0;
    for (int layer = layer_rng.u_; layer >= layer_rng.l_; --layer) {
      if (neighbor_cnt >= iparam_.M_) {
        break;
      }
      auto ll    = linklistsmemory_ + id * elemperlinklist_ + layer * (iparam_.M_ + 1);
      auto ll_sz = ll[iparam_.M_];
#ifdef USE_SSE
      _mm_prefetch((char *)(storage_->GetBaseVecByID(ll[0])), _MM_HINT_T0);
      _mm_prefetch((char *)(ll + 1), _MM_HINT_T0);
#endif
      bool visit_next_layer = false;
      for (int i = 0; i < ll_sz; ++i) {
        if (neighbor_cnt >= iparam_.M_) {
          break;
        }
        auto nn_id   = ll[i];
        if (nn_id < filter.l_ || nn_id > filter.u_) {
          visit_next_layer = true;
          continue;
        }

#ifdef USE_SSE
        _mm_prefetch((char *)(storage_->GetBaseVecByID(ll[i + 1])), _MM_HINT_T0);
#endif
        if (visited->Test(nn_id)) {
          continue;
        }
        visited->Set(nn_id);
        auto nn_dist = fstdistfunc_(v, storage_->GetBaseVecByID(nn_id), dist_func_param_);
        status.dist_computation_++;
        neighbor_cnt++;
        if (result.size() < ef || nn_dist < res_max_dist) {
          PUSH_HEAP(candidates, -nn_dist, nn_id);
          PUSH_HEAP(result, nn_dist, nn_id);
          if (result.size() > ef) {
            POP_HEAP(result);
          }
          res_max_dist = TOP_HEAP(result).dist_;
        }
      }
      if (!is_build && !visit_next_layer) {
        break;
      }
    }
    if (is_build)
      linklist_locks_[id].unlock();
  }
  visited_pool_.Return(visited);
  return result;
}

}  // namespace spatt