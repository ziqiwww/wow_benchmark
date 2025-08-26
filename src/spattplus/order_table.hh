

#pragma once

#include <vector>
#include "ygg.hpp"

#include "common/micro.hh"
#include "common/type.hh"
#include "common/dataloader.hh"
#include "common/disk.hh"

namespace spatt {
class OrderTable
{

public:
  OrderTable() = default;

  ~OrderTable() = default;

  DISABLE_COPY_MOVE_AND_ASSIGN(OrderTable);

  virtual void InsertLabel(label_t label) = 0;

  virtual auto GetWindowedFilterAndEntries(label_t cur_mata, int half_window_size, std::vector<label_t> &entry_points)
      -> QueryFilter<label_t> = 0;

  virtual auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates, label_t cur_id, int half_window_size)
      -> std::vector<dist_id_pair> = 0;

  virtual auto GetRangeCardinality(label_t l, label_t u) -> size_t { return u - l + 1; }

  virtual void Serialize(std::ostream &os) = 0;

  virtual void Deserialize(std::istream &is) = 0;

public:
  std::mutex lock_{};
};

class VectorOrderTable : public OrderTable
{

public:
  VectorOrderTable() = default;

  ~VectorOrderTable() = default;

  DISABLE_COPY_MOVE_AND_ASSIGN(VectorOrderTable);

  inline void InsertLabel(label_t label) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    auto                        it = GetMetaItr(label);
    // insert the label into vector
    order_.insert(it, label);
  }

  auto GetWindowedFilterAndEntries(label_t cur_mata, int half_window_size, std::vector<label_t> &entry_points)
      -> QueryFilter<label_t> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    QueryFilter<label_t>        filter;
    // locate the position of the current meta and get the windowed filter, the meta has not been inserted
    auto it                                 = GetMetaItr(cur_mata);
    int  pos_c                              = std::distance(order_.begin(), it);
    auto pos_l                              = std::max(0, pos_c - half_window_size);
    auto pos_u                              = std::max(0, std::min((int)order_.size() - 1, pos_c + half_window_size));
    filter.l_                               = order_[pos_l];
    filter.u_                               = order_[pos_u];
    unsigned                           seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine         generator(seed);
    std::uniform_int_distribution<int> distribution(pos_l, pos_u);
    for (int i = 0; i < std::min(pos_u - pos_l + 1, 3); ++i) {
      auto idx = distribution(generator);
      // if the idx is not in the entry_points, insert it
      if (std::find(entry_points.begin(), entry_points.end(), order_[idx]) == entry_points.end()) {
        entry_points.emplace_back(order_[idx]);
      }
    }
    return filter;
  }

  auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates, label_t cur_id, int half_window_size)
      -> std::vector<dist_id_pair> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    std::vector<dist_id_pair>   in_window_ids;
    // get the out of window id
    auto it       = GetMetaItr(cur_id);
    int  pos_c    = std::distance(order_.begin(), it);
    int  pos_l    = std::max(0, pos_c - half_window_size);
    int  pos_u    = std::max(0, std::min((int)order_.size() - 1, pos_c + half_window_size));
    auto filter_l = order_[pos_l];
    auto filter_u = order_[pos_u];
    for (auto &c : candidates) {
      if (c.id_ >= filter_l && c.id_ <= filter_u) {
        in_window_ids.emplace_back(c);
      }
    }
    return in_window_ids;
  }

  void Serialize(std::ostream &os) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // serialize the order table
    WriteBinaryPOD(os, order_.size());
    for (auto &m : order_) {
      WriteBinaryPOD(os, m);
    }
  }

  void Deserialize(std::istream &is) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // deserialize the order table
    size_t size;
    ReadBinaryPOD(is, size);
    order_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      ReadBinaryPOD(is, order_[i]);
    }
  }

private:
  inline auto GetMetaItr(label_t meta) -> std::vector<label_t>::iterator
  {
    // get the iterator of the meta in the order table
    auto it = std::lower_bound(order_.begin(), order_.end(), meta, [this](label_t a, label_t b) { return a < b; });
    return it;
  }

public:
  std::vector<label_t> order_{};
};

class WBTreeOrderTable : public OrderTable
{
  using MyTreeOptions = ygg::TreeOptions<ygg::TreeFlags::MULTIPLE, ygg::TreeFlags::WBT_DELTA_NUMERATOR<3>,
      ygg::TreeFlags::WBT_DELTA_DENOMINATOR<1>, ygg::TreeFlags::WBT_GAMMA_NUMERATOR<2>,
      ygg::TreeFlags::WBT_GAMMA_DENOMINATOR<1>>;

  class WBNode : public ygg::WBTreeNodeBase<WBNode, MyTreeOptions>
  {
  public:
    WBNode() = default;
    explicit WBNode(label_t label) : label_(label) {}
    ~WBNode() = default;

    bool operator<(const WBNode &other) const { return this->label_ < other.label_; }

  public:
    label_t label_{};
  };

  using MyTree = ygg::WBTree<WBNode, ygg::WBDefaultNodeTraits, MyTreeOptions>;

public:
  WBTreeOrderTable() = delete;

  WBTreeOrderTable(size_t max_N) : max_N_(max_N) { node_store_ = new WBNode[max_N]; }

  ~WBTreeOrderTable()
  {
    if (node_store_ != nullptr) {
      delete[] node_store_;
    }
  }

  inline void InsertLabel(label_t label) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    auto                        node = &node_store_[tree_.size()];
    node->label_                     = label;
    tree_.insert(*node);
  }

  inline auto GetWindowedFilterAndEntries(label_t cur_mata, int half_window_size, std::vector<label_t> &entry_points)
      -> QueryFilter<label_t> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    QueryFilter<label_t>        filter;
    // if pos_l == 0 and pos_u == order_.size() - 1, the filter is the whole range just return
    if (2 * half_window_size >= tree_.size()) {
      filter.l_ = tree_.begin()->label_;
      filter.u_ = tree_.rbegin()->label_;
      entry_points.emplace_back(tree_.begin()->label_);
      return filter;
    }
    auto    it               = tree_.lower_bound(WBNode(cur_mata));
    WBNode *cur_node         = it == tree_.end() ? &*tree_.rbegin() : &*it;
    auto    boundary_indexes = GetWindowRangeLabel(cur_node, half_window_size);

    filter.l_ = boundary_indexes.l_;
    filter.u_ = boundary_indexes.u_;
    entry_points.emplace_back(boundary_indexes.l_);
    if (boundary_indexes.l_ != boundary_indexes.u_) {
      entry_points.emplace_back(boundary_indexes.u_);
    }
    return filter;
  }

  auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates, label_t cur_id, int half_window_size)
      -> std::vector<dist_id_pair> override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    std::vector<dist_id_pair>   in_window_ids;
    // get the out of window id
    if (2 * half_window_size >= tree_.size()) {
      for (auto &c : candidates) {
        in_window_ids.emplace_back(c);
      }
      return in_window_ids;
    }
    auto it = tree_.find(WBNode(cur_id));
    if (it == tree_.end()) {
      throw std::runtime_error("Current node not found");
    }
    auto boundary_indexes = GetWindowRangeLabel(&*it, half_window_size);
    for (auto &c : candidates) {
      auto l_meta = boundary_indexes.l_;
      auto u_meta = boundary_indexes.u_;
      if (c.id_ >= l_meta && c.id_ <= u_meta) {
        in_window_ids.emplace_back(c);
      }
    }
    return in_window_ids;
  }

  auto GetNodeIndex(WBNode *root, label_t l) -> size_t
  {
    WBNode *cur = root;
    if (cur == nullptr)
      return 0;
    size_t index = 0;
    while (cur) {
      if (l < cur->label_) {
        cur = cur->get_left();
      } else if (l == cur->label_) {
        size_t left_size = cur->get_left() ? cur->get_left()->_wbt_size - 1 : 0;
        return index + left_size;
      } else {
        size_t left_size = cur->get_left() ? cur->get_left()->_wbt_size - 1 : 0;
        index += left_size + 1;
        cur = cur->get_right();
      }
    }
    throw std::runtime_error("Current node index not found");
  }

  auto FindUpperBound(WBNode *root, label_t target) -> label_t
  {
    WBNode *candidate = nullptr;
    WBNode *cur       = root;

    while (cur) {
      if (cur->label_ >= target) {
        candidate = cur;              // possible answer
        cur       = cur->get_left();  // find smaller answer
      } else {
        cur = cur->get_right();
      }
    }
    if (!candidate)
      throw std::runtime_error("Target upper bound not exist");
    return candidate->label_;  // first node >= target
  }

  auto FindLowerBound(WBNode *root, label_t target) -> label_t
  {
    WBNode *candidate = nullptr;
    WBNode *cur       = root;

    while (cur) {
      if (cur->label_ <= target) {
        candidate = cur;               // possible answer
        cur       = cur->get_right();  // find larger answer
      } else {
        cur = cur->get_left();
      }
    }
    if (!candidate)
      throw std::runtime_error("Target lower bound not exist");
    return candidate->label_;  // first node <= target
  }

  auto GetRangeCardinality(label_t l, label_t u) -> size_t override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // TODO: get the range cardinality, get the index of l, u: i, j and return j - i + 1
    WBNode *root    = tree_.get_root();
    label_t label_l = FindUpperBound(root, l);
    label_t label_u = FindLowerBound(root, u);
    size_t  i       = GetNodeIndex(root, label_l);
    size_t  j       = GetNodeIndex(root, label_u);
    return j - i + 1;
  }

  void Serialize(std::ostream &os) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // serialize the order table
    size_t size = tree_.size();
    WriteBinaryPOD(os, size);
    for (auto &n : tree_) {
      WriteBinaryPOD(os, n.label_);
    }
  }

  void Deserialize(std::istream &is) override
  {
    std::lock_guard<std::mutex> lock(this->lock_);
    // deserialize the order table
    size_t size;
    ReadBinaryPOD(is, size);
    LOG(fmt::format("Loading WBT size: {}", size));
    if (size > max_N_) {
      throw std::runtime_error("The size of the tree is larger than the max_N");
    }
    for (size_t i = 0; i < size; ++i) {
      ReadBinaryPOD(is, node_store_[i].label_);
      tree_.insert(node_store_[i]);
    }
  }

private:
  auto GetKthSmallestNode(WBNode *root, int k) -> WBNode *
  {
    if (root == nullptr || k <= 0 || k > root->_wbt_size - 1) {
      return nullptr;
    }

    WBNode *current = root;
    while (current != nullptr) {
      int leftSize = current->get_left() ? current->get_left()->_wbt_size - 1 : 0;

      if (k == leftSize + 1) {
        return current;
      } else if (k <= leftSize) {
        current = current->get_left();
      } else {
        k       = k - leftSize - 1;
        current = current->get_right();
      }
    }

    return nullptr;
  }
  auto GetKthLargestNode(WBNode *root, int k) -> WBNode *
  {
    if (root == nullptr || k <= 0 || k > root->_wbt_size - 1) {
      return nullptr;
    }

    WBNode *current = root;
    while (current != nullptr) {
      int rightSize = current->get_right() ? current->get_right()->_wbt_size - 1 : 0;

      if (k == rightSize + 1) {
        return current;
      } else if (k <= rightSize) {
        current = current->get_right();
      } else {
        k       = k - rightSize - 1;
        current = current->get_left();
      }
    }

    return nullptr;
  }

  auto GetWindowRangeLabel(WBNode *cur_node, int half_window_size) -> QueryFilter<label_t>
  {
    // get the index of the left boundary
    QueryFilter<label_t> boundary_indexes;
    int                  k        = half_window_size;
    WBNode              *current  = cur_node;
    int                  leftSize = current->get_left() ? current->get_left()->_wbt_size - 1 : 0;
    if (k <= leftSize) {
      // The kth smallest node is in the left subtree
      WBNode *left_boundry = GetKthLargestNode(current->get_left(), k);
      boundary_indexes.l_  = left_boundry->label_;
    } else {
      // The kth smallest node may be in the parent path
      k -= leftSize;
      while (current) {
        WBNode *parent = current->get_parent();
        if (!parent) {
          // no parent to go back
          boundary_indexes.l_ = tree_.begin()->label_;
          break;
        }
        if (current == parent->get_right()) {
          if ((parent->get_left() ? parent->get_left()->_wbt_size - 1 : 0) + 1 >= k) {
            if (k == 1) {
              boundary_indexes.l_ = parent->label_;
              break;
            } else {
              WBNode *left_boundry = GetKthLargestNode(parent->get_left(), k - 1);
              boundary_indexes.l_  = left_boundry->label_;
              break;
            }
          } else {
            k -= (parent->get_left() ? parent->get_left()->_wbt_size - 1 : 0) + 1;
            current = parent;
          }
        } else {
          current = parent;
        }
      }
    }
    // get the index of the right boundary
    k             = half_window_size;
    current       = cur_node;
    int rightSize = current->get_right() ? current->get_right()->_wbt_size - 1 : 0;
    if (k <= rightSize) {
      WBNode *right_boundry = GetKthSmallestNode(current->get_right(), k);
      boundary_indexes.u_   = right_boundry->label_;
    } else {
      k -= rightSize;
      while (current) {
        WBNode *parent = current->get_parent();
        if (!parent) {
          boundary_indexes.u_ = tree_.rbegin()->label_;
          break;
        }
        if (current == parent->get_left()) {
          if ((parent->get_right() ? parent->get_right()->_wbt_size - 1 : 0) + 1 >= k) {
            if (k == 1) {
              boundary_indexes.u_ = parent->label_;
              break;
            } else {
              WBNode *right_boundry = GetKthSmallestNode(parent->get_right(), k - 1);
              boundary_indexes.u_   = right_boundry->label_;
              break;
            }
          } else {
            k -= (parent->get_right() ? parent->get_right()->_wbt_size - 1 : 0) + 1;
            current = parent;
          }
        } else {
          current = parent;
        }
      }
    }
    return boundary_indexes;
  }

  void PrintFormatedTree(WBNode *node, std::string prefix, bool is_left)
  {
    if (node == nullptr) {
      return;
    }
    std::cout << prefix;
    std::cout << (is_left ? "├──" : "└──");
    // print the value of the node
    std::cout << node->label_ << std::endl;
    // enter the next tree level - left and right branch
    PrintFormatedTree(node->get_left(), prefix + (is_left ? "│   " : "    "), true);
    PrintFormatedTree(node->get_right(), prefix + (is_left ? "│   " : "    "), false);
  }

public:
  size_t  max_N_{};
  MyTree  tree_{};
  WBNode *node_store_{nullptr};
};

}  // namespace spatt
