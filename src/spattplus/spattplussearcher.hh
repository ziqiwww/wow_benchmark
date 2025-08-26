#pragma once

#include "spattplusindex.hh"
#include <omp.h>

namespace spatt {

struct SearchStatus
{
  RuntimeStatus             rt_status_{};
  std::vector<dist_id_pair> result_{};
};

template <typename vec_t = float>
class SpattPlusSearcher : public SpattPlus<vec_t>
{
public:
  SpattPlusSearcher(DataLoader<vec_t> *dl, const std::string &space);

  auto Search(const vec_t *query_vec, const QueryFilter<label_t> &filter, SearchParameters &sparam) -> SearchStatus;

private:
  inline auto DecideLayerRange(const QueryFilter<label_t> &filter_rng) -> QueryFilter<int>;
};

template <typename vec_t>
SpattPlusSearcher<vec_t>::SpattPlusSearcher(DataLoader<vec_t> *dl, const std::string &space)
    : SpattPlus<vec_t>(dl, space)
{}

template <typename vec_t>
auto SpattPlusSearcher<vec_t>::Search(
    const vec_t *query_vec, const QueryFilter<label_t> &filter, SearchParameters &sparam) -> SearchStatus
{
  SearchStatus status;
  auto         start = std::chrono::high_resolution_clock::now();
  auto         eps   = std::vector<label_t>{(filter.l_ + filter.u_) / 2};
  if(sparam.is_dynamic_)
    sparam.layer_rng_ = DecideLayerRange(filter);
  auto result =
      this->template SearchCandidates<false>(eps, query_vec, filter, sparam.layer_rng_, sparam.efs_, status.rt_status_);
  while (result.size() > sparam.k_) {
    POP_HEAP(result);
  }
  auto end                    = std::chrono::high_resolution_clock::now();
  status.rt_status_.run_time_ = std::chrono::duration<double>(end - start).count();
  status.result_              = std::move(result);
  return status;
}

template <typename vec_t>
auto SpattPlusSearcher<vec_t>::DecideLayerRange(const QueryFilter<label_t> &filter_rng) -> QueryFilter<int>
{
  QueryFilter<int> new_layer_rng;

  /**
   * we can use WBT to decide filter length if the labels are not sequential:
   * 
   * filter_length = this->order_table_->GetRangeCardinality(filter_rng.l_, filter_rng.u_); 
   * 
   * but in benchmarking, we can use the filter length directly.
  */ 
  int                filter_length = filter_rng.u_ - filter_rng.l_ + 1;
  // find the largest layer that can be covered by the filter
  auto c_it = std::lower_bound(this->window_size_.begin(), this->window_size_.end(), filter_length);
  if (c_it == this->window_size_.end() || *c_it > filter_length) {
    c_it--;
  }
  LOG_DBG(fmt::format("Filter length: {}, window size: {}", filter_length, *c_it));
  ASSERT_MSG(*c_it <= filter_length, fmt::format("Invalid window size: {}, filter length: {}", *c_it, filter_length));
  int c_it_idx = std::distance(this->window_size_.begin(), c_it);
  ASSERT_MSG(c_it_idx >= 0 && c_it_idx <= this->iparam_.wp_, fmt::format("Invalid c_it_idx: {}", c_it_idx));
  if (c_it_idx == 0) {
    new_layer_rng.l_ = 0;
    new_layer_rng.u_ = c_it_idx + 1;
  } else if (c_it_idx == this->iparam_.wp_) {
    new_layer_rng.l_ = c_it_idx - 1;
    new_layer_rng.u_ = c_it_idx;
  } else {
    int c_l = c_it_idx - 1;
    int c_u = c_it_idx + 1;
    // find the largest fraction
    float frac_l = 1.0 * this->window_size_[c_l] / filter_length;
    float frac_u = 1.0 * filter_length / std::min((int)this->window_size_[c_u], (int)this->iparam_.max_N_);
    if (frac_l > frac_u) {
      new_layer_rng.l_ = c_l;
      new_layer_rng.u_ = c_it_idx;
    } else {
      new_layer_rng.l_ = c_it_idx;
      new_layer_rng.u_ = c_u;
    }
  }
  new_layer_rng.l_ = 0;
  return new_layer_rng;
}

}  // namespace spatt