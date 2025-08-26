#pragma once
#include <algorithm>
#include "fmt/format.h"
#include "hnswlib/hnswalg.h"

namespace spatt {
typedef hnswlib::tableint label_t;
typedef uint32_t          graph_id_t;

#define ENUM_ENTITIES \
  ENUM(d_lin)         \
  ENUM(d_exp)         \
  ENUM(d_uni)
#define ENUM(x) ENUMENTRY(x)
DECLARE_ENUM(DecayMetric, ENUM_ENTITIES)
#undef ENUM
#define ENUM(x) STRING2ENUM(x)
STRING_TO_ENUM_BODY(DecayMetric, ENUM_ENTITIES)
#undef ENUM
#define ENUM(x) ENUM2STRING(x)
ENUM_TO_STRING_BODY(DecayMetric, ENUM_ENTITIES)
#undef ENUM
#undef ENUM_ENTITIES

#define ENUM_ENTITIES \
  ENUM(p_oneround)    \
  ENUM(p_seperate)
#define ENUM(x) ENUMENTRY(x)
DECLARE_ENUM(PruneMetric, ENUM_ENTITIES)
#undef ENUM
#define ENUM(x) STRING2ENUM(x)
STRING_TO_ENUM_BODY(PruneMetric, ENUM_ENTITIES)
#undef ENUM
#define ENUM(x) ENUM2STRING(x)
ENUM_TO_STRING_BODY(PruneMetric, ENUM_ENTITIES)
#undef ENUM
#undef ENUM_ENTITIES

typedef float dist_t;

template <typename att_t = int>
struct QueryFilter
{
  att_t l_{};
  att_t u_{};
};

struct dist_id_pair
{
  dist_t  dist_;
  label_t id_;

  dist_id_pair() = default;

  dist_id_pair(dist_t dist, label_t id) : dist_(dist), id_(id) {}

  bool operator<(const dist_id_pair &rhs) const { return dist_ < rhs.dist_; }

  bool operator>(const dist_id_pair &rhs) const { return dist_ > rhs.dist_; }
};

struct BuildParametersRuntime
{
  // beam search size during construction
  size_t efc_;
  /// @brief index unrelated parameters
  int threads_{1};
};

struct SearchParameters
{
  bool is_dynamic_;
  size_t efs_;
  size_t k_;
  // if layer_rng_.l_ > max_layer, this is a search that we should automatically decide the layer range
  QueryFilter<int> layer_rng_;
};

template <typename T>
inline auto VecToString(const std::vector<T> &vec) -> std::string
{
  std::string res;
  res += fmt::format("[size: {} | ", vec.size());
  for (const auto &v : vec) {
    res += fmt::format("{}, ", v);
  }
  res += "]";
  return res;
}

template <typename p1, typename p2>
inline auto PairVecToString(const std::vector<std::pair<p1, p2>> &vec) -> std::string
{
  std::string res;
  res += fmt::format("[size: {} | ", vec.size());
  for (const auto &p : vec) {
    res += fmt::format("({}, {}), ", p.first, p.second);
  }
  res += "]";
  return res;
}

inline auto DistIDVecToString(const std::vector<dist_id_pair> &vec) -> std::string
{
  std::string res;
  res += fmt::format("[size: {} | ", vec.size());
  for (const auto &p : vec) {
    res += fmt::format("({}, {}), ", p.dist_, p.id_);
  }
  res += "]";
  return res;
}

class VisitedBaseClass
{
public:
  virtual void Clear()                          = 0;
  virtual void ClearRange(label_t l, label_t u) = 0;
  virtual void Set(label_t i)                   = 0;
  virtual bool Test(label_t i)                  = 0;
  virtual void Reset(label_t i)                 = 0;
  virtual ~VisitedBaseClass()                   = default;
};

// memory optimized Bitset, aligned to 64 bits and use aligned alloc for cache line alignment
class Bitset : public VisitedBaseClass
{
public:
  Bitset(size_t n) : n_(n)
  {
    size_t n_bytes = (n + 7) / 8;
    data_          = static_cast<uint64_t *>(aligned_alloc(64, n_bytes));
    if (data_ == nullptr) {
      LOG("Failed to allocate memory for Bitset");
      std::abort();
    }
  }

  ~Bitset() override { free(data_); }
  DISABLE_COPY_MOVE_AND_ASSIGN(Bitset);

  inline void Set(label_t i) override { data_[i / 64] |= 1ULL << (i % 64); }

  inline bool Test(label_t i) override { return data_[i / 64] & (1ULL << (i % 64)); }

  inline void Reset(label_t i) override { data_[i / 64] &= ~(1ULL << (i % 64)); }

  inline auto GetData(label_t i) -> uint64_t * { return &data_[i / 64]; }

  inline void Clear() override
  {
    size_t n_bytes = (n_ + 7) / 8;
    memset(data_, 0, n_bytes);
  }

  inline void ClearRange(label_t l, label_t u) override
  {
    auto start_block = l / 64;
    auto end_block   = u / 64;
    memset(data_ + start_block, 0, (end_block - start_block + 1) * sizeof(uint64_t));
  }

public:
  size_t    n_{};
  uint64_t *data_{nullptr};
};

class VisitedList : public VisitedBaseClass
{
public:
  VisitedList(int numelements1)
  {
    curV_        = -1;
    numelements_ = numelements1;
    mass_        = static_cast<hnswlib::vl_type *>(aligned_alloc(64, numelements_ * sizeof(hnswlib::vl_type)));
  }

  inline void Clear() override
  {
    curV_++;
    if (curV_ == 0) {
      memset(mass_, 0, sizeof(hnswlib::vl_type) * numelements_);
      curV_++;
    }
  }

  inline void ClearRange(label_t l, label_t u) override { Clear(); }

  inline void Set(label_t i) override { mass_[i] = curV_; }

  inline bool Test(label_t i) override { return (mass_[i] == curV_); }

  inline void Reset(label_t i) override { mass_[i] = -1; }

  inline auto GetData(label_t i) -> hnswlib::vl_type * { return &mass_[i]; }

  ~VisitedList() override { free(mass_); }

public:
  hnswlib::vl_type  curV_;
  hnswlib::vl_type *mass_;
  unsigned int      numelements_;
};

template <typename VisitedType = Bitset>
class VisitedPool
{
public:
  /**
   * @brief Construct a new Visited Bit Set Pool object
   *
   * @param n number of elements for each bitset to store
   * @param pool_size number of bitset to store
   */
  VisitedPool() = default;

  ~VisitedPool()
  {
    for (auto bs : pool_) {
      delete bs;
    }
  }

  void Init(size_t n) { n_ = n; }

  DISABLE_COPY_MOVE_AND_ASSIGN(VisitedPool);

  inline auto Get() -> VisitedType *
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pool_.empty()) {
      return new VisitedType(n_);
    }
    auto bs = pool_.back();
    pool_.pop_back();
    return bs;
  }

  inline void Return(VisitedType *bs)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    pool_.push_back(bs);
  }

private:
  size_t                     n_{};
  std::vector<VisitedType *> pool_;
  std::mutex                 mtx_;
};

}  // namespace spatt