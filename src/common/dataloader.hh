#pragma once

#include <fstream>
#include <vector>
#include "micro.hh"
#include "type.hh"
namespace spatt {
template <typename vec_t = float>
class DataLoader
{
public:
  explicit DataLoader(
      const std::string &db_name, const std::string &v_type = "fvecs", const std::string &a_type = "int")
      : db_name_(db_name), v_type_(v_type), a_type_(a_type)
  {
    LOG(fmt::format("DataLoader for {}", db_name_));
    LOG(fmt::format("Vector type: {}, Attribute type: {}", v_type_, a_type_));
  }
  ~DataLoader()
  {
    if (xb_) {
      free(xb_);
    }
    if (xq_) {
      free(xq_);
    }
    if(base_meta_) {
      free(base_meta_);
    }
  }
  DISABLE_COPY_MOVE_AND_ASSIGN(DataLoader);

  /**
   * @brief fvecs is a binary file format for storing float vectors.
   * format:
   * d(4 bytes)
   * vector content: sizeof(dist_t) * d
   * total size: n * (4 + sizeof(dist_t) * d)
   *
   * @param fvecs_file
   */
  void Loadfves(
      const std::string &fvecs_file, bool is_query = false, size_t max_n = std::numeric_limits<size_t>::max());

  /**
   * @brief meta file is a binary file with fixed size of att_t
   *
   * @param meta_file
   */
  void LoadBaseMeta(const std::string &meta_file){
    // meta file is a binary file with nb * sizeof(int) bytes
    if(meta_file=="null") {
      LOG("Meta file is empty");
      std::abort();
    }
    std::ifstream ifs(meta_file, std::ios::binary);
    if (!ifs.is_open()) {
      LOG(fmt::format("Failed to open file: {}", meta_file));
      std::abort();
    }
    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t n_meta = file_size / sizeof(int);
    if(nb_ != n_meta) {
      LOG(fmt::format("Base meta size: {} is not equal to base size: {}", n_meta, nb_));
      std::abort();
    }
    base_meta_ = (int *)malloc(n_meta * sizeof(int));
    if (base_meta_ == nullptr) {
      LOG(fmt::format("Not enough memory for base meta"));
      std::abort();
    }
    ifs.read(reinterpret_cast<char *>(base_meta_), static_cast<std::streamsize>(n_meta * sizeof(int)));
    ifs.close();
    LOG(fmt::format("Base meta: Num: {}, Size: {}", n_meta, sizeof(int)));
    return;
  }

  /**
   * @brief query filter is 2 * sizeof(att_t) * nq sized binary file
   *
   * @param query_filter_file
   */
  void LoadQueryFilter(const std::string &query_filter_file);

  inline auto GetBaseVecByID(size_t id) const -> const vec_t * { return xb_ + id * d_; }

  inline auto GetQueryVecByID(size_t id) const -> const vec_t * { return xq_ + id * d_; }

  inline auto GetQueryFilterByID(size_t id) const -> const QueryFilter<label_t> & { return query_filters_[id]; }

public:
  std::string db_name_;
  std::string v_type_;
  std::string a_type_;
  size_t      d_{};
  size_t      nb_{};
  size_t      nq_{};
  vec_t      *xb_{nullptr};
  vec_t      *xq_{nullptr};
  int        *base_meta_{nullptr};
  size_t      meta_cardinality_{};  // number of unique meta, used for checking w
  // query range, if first == second then it is a single query
  std::vector<QueryFilter<label_t>> query_filters_{};
};

template <typename vec_t>
inline void DataLoader<vec_t>::Loadfves(const std::string &fvecs_file, bool is_query, size_t max_n)
{
  std::ifstream ifs(fvecs_file, std::ios::binary);
  if (!ifs.is_open()) {
    LOG(fmt::format("Failed to open file: {}", fvecs_file));
    std::abort();
  }
  // calculate the number of vectors
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  size_t v_size = 4;
  // read the first d
  ifs.read(reinterpret_cast<char *>(&d_), static_cast<std::streamsize>(v_size));
  size_t v_content_size = d_ * sizeof(vec_t);
  ifs.seekg(0, std::ios::beg);
  if (is_query) {
    nq_ = file_size / (v_size + v_content_size);
    nq_ = std::min(nq_, max_n);
#ifdef ALIGNVECTORS
    xq_ = (vec_t *)aligned_alloc(64, nq_ * d_ * sizeof(vec_t));
#else
    xq_ = (vec_t *)malloc(nq_ * d_ * sizeof(vec_t));
#endif
    if (xq_ == nullptr) {
      LOG(fmt::format("Not enough memory for xq"));
      std::abort();
    }
    // each vector has 4 byte d and d * sizeof(vec_t) content
    for (size_t i = 0; i < nq_; ++i) {
      ifs.read(reinterpret_cast<char *>(&d_), static_cast<std::streamsize>(v_size));
      ifs.read(reinterpret_cast<char *>(xq_ + i * d_), static_cast<std::streamsize>(v_content_size));
    }
  } else {
    nb_ = file_size / (v_size + v_content_size);
    nb_ = std::min(nb_, max_n);
#ifdef ALIGNVECTORS
    xb_ = (vec_t *)aligned_alloc(64, nb_ * d_ * sizeof(vec_t));
#else
    xb_ = (vec_t *)malloc(nb_ * d_ * sizeof(vec_t));
#endif
    if (xb_ == nullptr) {
      LOG(fmt::format("Not enough memory for xb"));
      std::abort();
    }
    for (size_t i = 0; i < nb_; ++i) {
      ifs.read(reinterpret_cast<char *>(&d_), static_cast<std::streamsize>(v_size));
      ifs.read(reinterpret_cast<char *>(xb_ + i * d_), static_cast<std::streamsize>(v_content_size));
    }
  }
  ifs.close();
  if (is_query) {
    LOG(fmt::format("Query: Dim: {}, Num: {}", d_, nq_));
  } else {
    LOG(fmt::format("Base: Dim: {}, Num: {}", d_, nb_));
  }
}

template <typename vec_t>
inline void DataLoader<vec_t>::LoadQueryFilter(const std::string &query_filter_file)
{
  std::ifstream ifs(query_filter_file, std::ios::binary);
  if (!ifs.is_open()) {
    LOG(fmt::format("Failed to open file: {}", query_filter_file));
    std::abort();
  }
  // check meta size
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto n_query = file_size / (2 * sizeof(int));
  if (n_query != nq_) {
    nq_ = std::min(nq_, n_query);
    LOG(fmt::format("Query filter size: {} is not equal to query size: {}", n_query, nq_));
  }
  query_filters_.resize(nq_);
  for (size_t i = 0; i < nq_; ++i) {
    int l, u;
    ifs.read(reinterpret_cast<char *>(&l), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&u), sizeof(int));
    query_filters_[i] = QueryFilter<label_t>{static_cast<label_t>(l), static_cast<label_t>(u)};
  }
  ifs.close();
  LOG(fmt::format("{} query filters loaded", nq_));
  // LOG first 10 query filters
  std::string qf_str;
  for (size_t i = 0; i < std::min<size_t>(10, nq_); ++i) {
    qf_str += fmt::format("({}, {}), ", query_filters_[i].l_, query_filters_[i].u_);
  }
  LOG(fmt::format("First 10 query filters: {}", qf_str));
}

}  // namespace spatt
