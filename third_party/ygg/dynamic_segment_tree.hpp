//
// Created by lukas on 10.11.17.
//

#ifndef YGG_DYNAMIC_SEGMENT_TREE_HPP
#define YGG_DYNAMIC_SEGMENT_TREE_HPP

#include "benchmark_sequence.hpp"
#include "debug.hpp"
#include "options.hpp"
#include "rbtree.hpp"
#include "size_holder.hpp"
#include "util.hpp"
#include "wbtree.hpp"
#include "ziptree.hpp"

#include <algorithm>
#include <type_traits>

namespace ygg {

// Forwards
template <class Node, class NodeTraits, class Combiners, class Options,
          class TreeSelector, class Tag>
class DynamicSegmentTree;
template <class KeyT, class AggValueT, class... Combiners>
class CombinerPack;

namespace dyn_segtree_internal {

template <class T>
constexpr bool
noexcept_math()
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
	return std::is_nothrow_constructible_v<T> &&
	       std::is_nothrow_assignable_v<std::add_lvalue_reference_t<T>,
	                                    T> && // Creation & assignment
	       noexcept(std::declval<T>() + std::declval<T>()) && noexcept(
	           std::declval<T>() - std::declval<T>()) && // basic arithmetic
	       noexcept(std::declval<T>() * std::declval<T>()) && noexcept(
	           std::declval<T>() / std::declval<T>()) && //
	       noexcept(std::declval<T>() > std::declval<T>()) && noexcept(
	           std::declval<T>() >= std::declval<T>()) && // comparison
	       noexcept(std::declval<T>() < std::declval<T>()) && noexcept(
	           std::declval<T>() <= std::declval<T>()) && //
	       noexcept(std::declval<T>() == std::declval<T>()) && noexcept(
	           std::declval<T>() != std::declval<T>());
#pragma GCC diagnostic pop
}

template <class T1, class T2>
constexpr bool
noexcept_math_ordered_pair()
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
	return noexcept(std::declval<T1>() * std::declval<T2>()) && noexcept(
	    std::declval<T1>() / std::declval<T2>()) && // basic arithmetic
	    noexcept(std::declval<T1>() > std::declval<T2>()) && noexcept(
	        std::declval<T1>() >= std::declval<T2>()) && // comparison
	    noexcept(std::declval<T1>() < std::declval<T2>()) && noexcept(
	        std::declval<T1>() <= std::declval<T2>()) && //
	    noexcept(std::declval<T1>() == std::declval<T2>()) && noexcept(
	        std::declval<T1>() != std::declval<T2>());
#pragma GCC diagnostic pop
}

template <class T1, class T2>
constexpr bool
noexcept_math()
{
	return noexcept_math<T1>() && noexcept_math<T2>() &&
	       noexcept_math_ordered_pair<T1, T2>() &&
	       noexcept_math_ordered_pair<T2, T1>();
}

template <class AggValueT, class... Combiners>
constexpr bool
noexcept_all_combiners()
{
	return (... && noexcept_math<typename Combiners::ValueT>()) &&
	       (... && noexcept_math<AggValueT, typename Combiners::ValueT>());
}

/* noexcept_dst* checks if all operations needed by the DST are noexcept.
 * This especially concerns operations on the combiners.
 */
template <class CombinerPackParam>
struct noexcept_dst_impl
{
};

template <template <class, class, class...> class CombinerPackParam, class KeyT,
          class AggValueT, class... Combiners>
struct noexcept_dst_impl<CombinerPackParam<KeyT, AggValueT, Combiners...>>
{
	static constexpr bool
	get()
	{
		return noexcept_all_combiners<AggValueT, Combiners...>();
	}
};

template <class ValueT, class CombinerPackParam>
constexpr bool
noexcept_dst()
{
	return noexcept_dst_impl<CombinerPackParam>::get() && noexcept_math<ValueT>();
}

/* Interface for when modification sequences of the underlying BST should be
 * stored for benchmarking purposes */
template <class InnerNode, class KeyT_in>
class InnerSequenceInterface {
public:
	using KeyT = KeyT_in;

	static KeyT_in
	get_key(const InnerNode & n) noexcept
	{
		return n.get_point();
	}

	static KeyT_in
	get_key(const KeyT_in query) noexcept
	{
		return query;
	}

	KeyT_in
	get_key(const std::pair<KeyT_in, bool> & query) noexcept
	{
		return std::get<0>(query);
	}
};
/// @cond INTERNAL

// Forwards
template <class InnerNode>
class Compare;
template <class InnerTree, class InnerNode, class Node, class NodeTraits>
class InnerRBNodeTraits;
template <class InnerTree, class InnerNode, class AggValueT>
class InnerZNodeTraits;
template <class InnerTree, class InnerNode, class Node, class AggValueT>
class InnerWBNodeTraits;

template <class Tag>
class InnerRBTTag {
};

template <class Tag>
class InnerZTTag {
};

template <class Tag>
class InnerWBTTag {
};

/********************************************
 * Base Class Definitions for RBTree
 ********************************************/
template <class... AdditionalOptions>
struct UseRBTree
{
	template <class InnerNode, class KeyT>
	using Options = TreeOptions<TreeFlags::MULTIPLE,
	                            TreeFlags::BENCHMARK_SEQUENCE_INTERFACE<
	                                InnerSequenceInterface<InnerNode, KeyT>>,
	                            AdditionalOptions...>;

	template <class Tag>
	struct InnerNodeBaseBuilder
	{
		template <class InnerNodeCRTP, class KeyT>
		using Base =
		    RBTreeNodeBase<InnerNodeCRTP, Options<InnerNodeCRTP, KeyT>, Tag>;
	};

	template <class CRTP, class Node, class NodeTraits, class InnerNode,
	          class Tag>
	using BaseTree =
	    RBTree<InnerNode,
	           dyn_segtree_internal::InnerRBNodeTraits<CRTP, InnerNode, Node,
	                                                   NodeTraits>,
	           Options<InnerNode, typename InnerNode::KeyT>,
	           dyn_segtree_internal::InnerRBTTag<Tag>, Compare<InnerNode>>;

	template <class TagType>
	using Tag = InnerRBTTag<TagType>;
};

/********************************************
 * Base Class Definitions for WBTree
 ********************************************/
template <class... AdditionalOptions>
struct UseWBTree
{
	template <class InnerNode, class KeyT>
	using Options = TreeOptions<TreeFlags::MULTIPLE, TreeFlags::WBT_SINGLE_PASS,
	                            TreeFlags::BENCHMARK_SEQUENCE_INTERFACE<
	                                InnerSequenceInterface<InnerNode, KeyT>>,
	                            AdditionalOptions...>;

	template <class Tag>
	struct InnerNodeBaseBuilder
	{
		template <class InnerNodeCRTP, class KeyT>
		using Base = WBTreeNodeBase<InnerNodeCRTP, Options<InnerNodeCRTP, KeyT>,
		                            Tag>; // TODO make options passable here
	};

	template <class CRTP, class Node, class NodeTraits, class InnerNode,
	          class Tag>
	using BaseTree =
	    WBTree<InnerNode,
	           dyn_segtree_internal::InnerWBNodeTraits<CRTP, InnerNode, Node,
	                                                   NodeTraits>,
	           Options<InnerNode, typename InnerNode::KeyT>,
	           dyn_segtree_internal::InnerWBTTag<Tag>, Compare<InnerNode>>;

	template <class TagType>
	using Tag = InnerWBTTag<TagType>;
};

/********************************************
 * Base Class Definitions for Zip Tree
 ********************************************/

template <class... AdditionalOptions>
struct UseZipTree
{
	template <class InnerNode>
	struct ZipTreeHasher
	{
		size_t
		operator()(const InnerNode & n) const noexcept
		{
			return std::hash<size_t>()(reinterpret_cast<size_t>(&n));
		}
	};

	template <class InnerNode, class KeyT>
	using Options =
	    TreeOptions<TreeFlags::MULTIPLE,
	                TreeFlags::BENCHMARK_SEQUENCE_INTERFACE<
	                    InnerSequenceInterface<InnerNode, KeyT>>,
	                TreeFlags::ZTREE_HASHER_TYPE<ZipTreeHasher<InnerNode>>,
	                AdditionalOptions...>;

	template <class Tag>
	struct InnerNodeBaseBuilder
	{
		template <class InnerNodeCRTP, class KeyT>
		class Base : public ZTreeNodeBase<InnerNodeCRTP,
		                                  Options<InnerNodeCRTP, KeyT>, Tag> {
		public:
			using MyBase =
			    ZTreeNodeBase<InnerNodeCRTP, Options<InnerNodeCRTP, KeyT>, Tag>;

			// Export to public, so that we can update ranks
			void
			update_rank() noexcept
			{
				this->MyBase::update_rank();
			}
		};
	};

	template <class CRTP, class Node, class NodeTraits, class InnerNode,
	          class Tag>
	using BaseTree =
	    ZTree<InnerNode,
	          dyn_segtree_internal::InnerZNodeTraits<
	              CRTP, InnerNode, typename InnerNode::AggValueT>,
	          Options<InnerNode, typename InnerNode::KeyT>,
	          dyn_segtree_internal::InnerZTTag<Tag>, Compare<InnerNode>>;

	template <class TagType>
	using Tag = InnerZTTag<TagType>;
};

/// @endcond

/**
 * @brief Representation of either a start or an end of an interval
 *
 * An object of this class represents either a start or an end of an interval
 * you inserted into a DynamicSegmentTree. You can use get_interval() to
 * retrieve a pointer to the node that you inserted into the DynamicSegmentTree.
 */
template <template <class InnerNodeCRTP, class BaseKeyT> class Base,
          class OuterNode, class KeyT_in, class ValueT_in, class AggValueT_in,
          class Combiners, class Tag>
class InnerNode : public Base<InnerNode<Base, OuterNode, KeyT_in, ValueT_in,
                                        AggValueT_in, Combiners, Tag>,
                              KeyT_in> {
public:
	/**
	 * @brief The type of the key (i.e., the interval bounds)
	 */
	using KeyT = KeyT_in;
	/**
	 * @brief The type of the value associated with the intervals
	 */
	using ValueT = ValueT_in;
	/**
	 * @brief The type of the aggregate value
	 */
	using AggValueT = AggValueT_in;

	/**
	 * @brief Returns the point at which the event represented by this InnerNode
	 * happens
	 *
	 * @return The point at which the event represented by this InnerNode happens
	 */
	KeyT get_point() const noexcept;

	/**
	 * @brief Returns true if this InnerNode represents an interval start
	 *
	 * @return true if this InnerNode represents an interval start
	 */
	bool is_start() const noexcept;

	/**
	 * @brief Returns true if this InnerNode represents an interval end
	 *
	 * @return true if this InnerNode represents an interval end
	 */
	bool is_end() const noexcept;

	/**
	 * @brief Returns true if the interval border represented by this InnerNode is
	 * closed
	 *
	 * @return true if the interval border represented by this InnerNode is closed
	 */
	bool is_closed() const noexcept;

	/**
	 * @brief Returns a pointer to your interval node
	 *
	 * This returns a pointer to an DynSegTreeNodeBase, which is the base class
	 * from which you have derived your Node class. You can up-cast this into your
	 * node class to get a pointer to the interval node.
	 *
	 * @return a pointer to your interval node
	 */
	const OuterNode * get_interval() const noexcept;

	// TODO this must be reset when inserting into the tree!
	size_t lca_tag = 0;

private:
	// TODO instead of storing all of these use interval traits and container
	// pointer?
	KeyT point;
	bool start;
	bool closed;

	// TODO remove this
	OuterNode * container;

	AggValueT agg_left;
	AggValueT agg_right;

	Combiners combiners;

	// The tree and the node traits have full access to the nodes
	template <class FNode, class FNodeTraits, class FCombiners, class FOptions,
	          class TreeSelector, class FTag>
	friend class ::ygg::DynamicSegmentTree;
	template <class FInnerTree, class FInnerNode, class FNode, class FNodeTraits>
	friend class InnerRBNodeTraits;
	template <class FInnerTree, class FInnerNode, class FNode, class FNodeTraits>
	friend class InnerWBNodeTraits;
	template <class FInnerTree, class FInnerNode, class FAggValueT>
	friend class InnerZNodeTraits;

	// Also, debugging classes are friends
	template <class FInnerNode, class... FCombiners>
	friend class ASCIIInnerNodeNameGetter;
	template <class FInnerNode, class... FCombiners>
	friend class DOTInnerNodeNameGette;
};

/// @cond INTERNAL

template <class InnerTree, class InnerNode, class AggValueT>
class InnerZNodeTraits { // TODO inherit from default traits?
public:
	/*
	 * Data for Zipping
	 */
	AggValueT left_accumulated;
	AggValueT right_accumulated;

	/*
	 * Callbacks for Zipping
	 */
	void init_zipping(const InnerNode * to_be_deleted) noexcept;
	void before_zip_from_left(InnerNode * left_head) noexcept;
	void before_zip_from_right(InnerNode * right_head) noexcept;
	void before_zip_tree_from_left(InnerNode * left_head) const noexcept;
	void before_zip_tree_from_right(InnerNode * right_head) const noexcept;
	void
	zipping_ended_left_without_tree(InnerNode * prev_left_head) const noexcept;
	void
	zipping_ended_right_without_tree(InnerNode * prev_right_head) const noexcept;
	void zipping_done(InnerNode * head, InnerNode * tail) const noexcept;
	void delete_without_zipping(const InnerNode * to_be_deleted) const noexcept;

	/*
	 * Data for Unzipping
	 */
	InnerNode * unzip_left_last;
	InnerNode * unzip_right_last;
	bool unzip_left_first;
	bool unzip_right_first;

	void init_unzipping(InnerNode * to_be_inserted) noexcept;
	void unzip_to_left(InnerNode * n) noexcept;
	void unzip_to_right(InnerNode * n) noexcept;
	void unzip_done(InnerNode * unzip_root, InnerNode * left_spine_end,
	                InnerNode * right_spine_end) noexcept;
};

template <class InnerTree, class InnerNode, class Node, class NodeTraits>
class InnerRBNodeTraits : public RBDefaultNodeTraits {
public:
	template <class RBTreeBase>
	static void leaf_inserted(const InnerNode & node,
	                          const RBTreeBase & t) noexcept;
	template <class RBTreeBase>
	static void rotated_left(InnerNode & node, const RBTreeBase & t) noexcept;
	template <class RBTreeBase>
	static void rotated_right(InnerNode & node, const RBTreeBase & t) noexcept;
	template <class RBTreeBase>
	static void delete_leaf(const InnerNode & node,
	                        const RBTreeBase & t) noexcept;
	template <class RBTreeBase>
	static void swapped(InnerNode & n1, InnerNode & n2, RBTreeBase & t) noexcept;

private:
	static InnerNode * get_partner(const InnerNode & n) noexcept;
};

/*
 * Since weight-balanced trees are based on the very same rotations that
 * red-black trees are, we can use the exact same traits.
 */
template <class InnerTree, class InnerNode, class Node, class NodeTraits>
class InnerWBNodeTraits
    : public InnerRBNodeTraits<InnerTree, InnerNode, Node, NodeTraits> {
public:
	template <class WBTreeBase>
	static void splice_out_left_knee(
	    InnerNode & node,
	    const WBTreeBase & t) noexcept; // TODO if aggregate-addition is noexcept
	template <class WBTreeBase>
	static void
	splice_out_right_knee(InnerNode & node,
	                      const WBTreeBase & t) noexcept; // TODO see above
};

template <class InnerNode>
class Compare {
public:
	using PointDescription =
	    std::pair<const typename InnerNode::KeyT, const int_fast8_t>;

	bool operator()(const InnerNode & lhs, const InnerNode & rhs) const noexcept;
	bool operator()(const typename InnerNode::KeyT & lhs,
	                const InnerNode & rhs) const noexcept;
	bool operator()(const InnerNode & lhs,
	                const typename InnerNode::KeyT & rhs) const noexcept;
	bool operator()(const PointDescription & lhs,
	                const InnerNode & rhs) const noexcept;
	bool operator()(const InnerNode & lhs,
	                const PointDescription & rhs) const noexcept;
};

/*
 * Debugging helpers
 */
template <class InnerNode, class... Combiners>
class ASCIIInnerNodeNameGetter {
public:
	std::string get_name(InnerNode * node) const;
};

template <class InnerNode, class... Combiners>
class DOTInnerNodeNameGetter {
public:
	std::string get_name(InnerNode * node) const;
};

template <class InnerNode>
class DOTInnerEdgeNameGetter {
public:
	std::string get_name(InnerNode * node, bool left) const;
};

/// @endcond
} // namespace dyn_segtree_internal

/**
 * @brief Class used to select the red-black tree as underlying tree for the
 * DynamicSegmentTree
 *
 * Use this class as the TreeSelector template parameter of the
 * DynamicSegmentTree to chose a red-black tree (an RBTree) as underlying tree
 * for the DynamicSegmentTree.
 *
 * @tparam AdditionalOptions Pass additional TreeFlags to the underlying
 *         red-black tree.
 */
template <class... AdditionalOptions>
class UseRBTree : public dyn_segtree_internal::UseRBTree<AdditionalOptions...> {
};
using UseDefaultRBTree = UseRBTree<>;

/**
 * @brief Class used to select the Zip Tree tree as underlying tree for the
 * DynamicSegmentTree
 *
 * Use this class as the TreeSelector template parameter of the
 * DynamicSegmentTree to chose a ZipTree (an ZTree) as underlying tree for the
 * DynamicSegmentTree.
 *
 * @tparam AdditionalOptions Pass additional TreeFlags to the underlying
 *         zip tree.
 */
template <class... AdditionalOptions>
class UseZipTree
    : public dyn_segtree_internal::UseZipTree<AdditionalOptions...> {
};
using UseDefaultZipTree = UseZipTree<TreeFlags::ZTREE_RANK_TYPE<std::uint8_t>>;

/**
 * @brief Class used to select the weight balanced tree as underlying tree for
 * the DynamicSegmentTree
 *
 * Use this class as the TreeSelector template parameter of the
 * DynamicSegmentTree to chose a weight balanced tree (see WBTree) as underlying
 * tree for the DynamicSegmentTree.
 *
 * @tparam AdditionalOptions Pass additional TreeFlags to the underlying
 *         red-black tree.
 */
template <class... AdditionalOptions>
class UseWBTree : public dyn_segtree_internal::UseWBTree<AdditionalOptions...> {
};
using UseDefaultWBTree = UseWBTree<>;

/**
 * @brief A combiner that allows to retrieve the maximum value over any range
 *
 * This is a combiner (see TODO for what a combiner is) that, when added to a
 * Dynamic Segment Tree, allows you to efficiently retrieve the maximum
 * aggregate value over any range in your segment tree.
 *
 * @tparam KeyType	 The type of the interval borders
 * @tparam ValueType The type of values associated with your intervals
 */
template <class KeyType, class ValueType>
class MaxCombiner {
public:
	using ValueT = ValueType;
	using KeyT = KeyType;
	using MyType = MaxCombiner<KeyT, ValueT>;

	MaxCombiner() = default;

	// TODO the bool is only returned for sake of expansion! Fix that!
	/**
	 * @brief Combines this MaxCombiner with a value, possibly of a child node
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the currently stored value and the value of left_child_combiner plus
	 * edge_val.
	 *
	 * Usually, a will be the value of the MaxCombiner of a child of the node that
	 * this combiner belongs to. edge_val will then be the agg_left value of the
	 * node this combiner belongs to.
	 *
	 * @param my_point 					  The point of the inner
	 * node that this MaxCombiner is associated with
	 * @param left_child_combiner The MaxCombiner belonging to the left child of
	 * this node
	 * @param edge_val 				    The aggregate value of the
	 * left edge going out of this node
	 * @return FIXME ignored for now
	 */
	bool collect_left(
	    const KeyT my_point, const MyType * left_child_combiner,
	    const ValueType
	        edge_val) noexcept(dyn_segtree_internal::noexcept_math<ValueT>());
	/**
	 * @brief Combines this MaxCombiner with a value, possibly of a child node
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the currently stored value and the value of right_child_combiner plus
	 * edge_val.
	 *
	 * Usually, a will be the value of the MaxCombiner of a child of the node that
	 * this combiner belongs to. edge_val will then be the agg_right value of the
	 * node this combiner belongs to.
	 *
	 * @param my_point 					   The point of the
	 * inner node that this MaxCombiner is associated with
	 * @param right_child_combiner The MaxCombiner belonging to the right child of
	 * this node
	 * @param edge_val 				     The aggregate value of the
	 * right edge going out of this node
	 * @return FIXME ignored for now
	 */
	// TODO make all keys / values references?
	bool collect_right(
	    const KeyT my_point, const MyType * right_child_combiner,
	    const ValueType
	        edge_val) noexcept(dyn_segtree_internal::noexcept_math<ValueT>());

	// TODO the bool is only returned for sake of expansion! Fix that!
	/**
	 * @brief Aggregates a value into the max value stored in this combiner
	 *
	 * This adds edge_val to the maximum currently stored in this combiner. This
	 * is used when traversing up a left edge in the tree.
	 *
	 * @param new_point 				The point of the node we
	 * traversed into
	 * @param edge_val 					The value of the edge we
	 * traversed
	 * @return FIXME ignored for now
	 */
	bool
	traverse_left_edge_up(const KeyT new_point, const ValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_math<ValueT>());
	/**
	 * @brief Aggregates a value into the max value stored in this combiner
	 *
	 * This adds edge_val to the maximum currently stored in this combiner. This
	 * is used when traversing up a right edge in the tree.
	 *
	 * @param new_point 				The point of the node we
	 * traversed into
	 * @param edge_val 					The value of the edge we
	 * traversed
	 * @return FIXME ignored for now
	 */
	bool
	traverse_right_edge_up(const KeyT new_point, const ValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_math<ValueT>());

	// bool aggregate_with(ValueT a);

	/**
	 * @brief Rebuilds the value in this MaxCombiner from values of its two
	 * children's MaxCombiners
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the left_child_combiner's value plus left_edge_val and
	 * right_child_combiner's value plus right_edge_val.
	 *
	 * @param my_point				       The point of the node
	 * this combiner belongs to
	 * @param left_child_combiner		 The MaxCombiner of the left child of
	 * this node
	 * @param left_edge_val					 The agg_left
	 * value of this node
	 * @param right_child_combiner	 The MaxCombiner of the right child of
	 * this node
	 * @param left_edge_val					 The agg_right
	 * value of this node
	 * @return FIXME ignored for now
	 */
	bool rebuild(KeyT my_point, const MyType * left_child_combiner,
	             ValueT left_edge_val, const MyType * right_child_combiner,
	             ValueT right_edge_val) noexcept(dyn_segtree_internal::
	                                                 noexcept_math<ValueT>());

	/**
	 * @brief Returns the currently stored combined value in this combiner
	 *
	 * @return the currently stored combined value in this combiner
	 */
	ValueT get() const noexcept;

	// TODO DEBUG
	static std::string
	get_name()
	{
		return "MaxCombiner";
	}
	// TODO DEBUG
	std::string
	get_dbg_value() const
	{
		return std::to_string(this->val);
	}

private:
	ValueT val;

	ValueT child_value(const MyType * child) const noexcept;
};

/**
 * @brief A combiner that allows to retrieve the maximum value over any range
 * plus the range over which the maximum occucrs.
 *
 * This is a combiner (see TODO for what a combiner is) that, when added to a
 * Dynamic Segment Tree, allows you to efficiently retrieve the maximum
 * aggregate value over any range in your segment tree. It will also tell you in
 * which range the maximum occurs.
 *
 * @tparam KeyType   The type of the interval borders
 * @tparam ValueType The type of values associated with your intervals
 */
template <class KeyType, class ValueType>
class RangedMaxCombiner {
public:
	using ValueT = ValueType;
	using KeyT = KeyType;
	using MyType = RangedMaxCombiner<KeyT, ValueT>;

	RangedMaxCombiner() noexcept;

	// TODO the bool is only returned for sake of expansion! Fix that!
	/**
	 * @brief Combines this RangedMaxCombiner with a value, possibly of a child
	 * node
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the currently stored value and the value of left_child_combiner plus
	 * edge_val.
	 *
	 * Usually, a will be the value of the RangedMaxCombiner of a child of the
	 * node that this combiner belongs to. edge_val will then be the agg_left
	 * value of the node this combiner belongs to.
	 *
	 * @param my_point 					  The point of the inner
	 * node that this RangedMaxCombiner is associated with
	 * @param left_child_combiner The RangedMaxCombiner belonging to the left
	 * child of this node
	 * @param edge_val 				    The aggregate value of the
	 * left edge going out of this node
	 * @return FIXME ignored for now
	 */
	bool collect_left(
	    KeyT my_point, const MyType * left_child_combiner,
	    ValueType
	        edge_val) noexcept(dyn_segtree_internal::noexcept_math<ValueT>());
	/**
	 * @brief Combines this RangedMaxCombiner with a value, possibly of a child
	 * node
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the currently stored value and the value of right_child_combiner plus
	 * edge_val.
	 *
	 * Usually, a will be the value of the RangedMaxCombiner of a child of the
	 * node that this combiner belongs to. edge_val will then be the agg_right
	 * value of the node this combiner belongs to.
	 *
	 * @param my_point 					   The point of the
	 * inner node that this RangedMaxCombiner is associated with
	 * @param right_child_combiner The RangedMaxCombiner belonging to the right
	 * child of this node
	 * @param edge_val 				     The aggregate value of the
	 * right edge going out of this node
	 * @return FIXME ignored for now
	 */
	bool collect_right(
	    KeyT my_point, const MyType * right_child_combiner,
	    ValueType
	        edge_val) noexcept(dyn_segtree_internal::noexcept_math<ValueT>());

	// TODO the bool is only returned for sake of expansion! Fix that!
	/**
	 * @brief Aggregates a value into the max value stored in this combiner
	 *
	 * This adds edge_val to the maximum currently stored in this combiner. This
	 * is used when traversing up a left edge in the tree.
	 *
	 * @param new_point 				The point of the node we
	 * traversed into
	 * @param edge_val 					The value of the edge we
	 * traversed
	 * @return FIXME ignored for now
	 */
	bool traverse_left_edge_up(KeyT new_point, ValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_math<ValueT>());
	/**
	 * @brief Aggregates a value into the max value stored in this combiner
	 *
	 * This adds edge_val to the maximum currently stored in this combiner. This
	 * is used when traversing up a right edge in the tree.
	 *
	 * @param new_point 				The point of the node we
	 * traversed into
	 * @param edge_val 					The value of the edge we
	 * traversed
	 * @return FIXME ignored for now
	 */
	bool traverse_right_edge_up(KeyT new_point, ValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_math<ValueT>());

	// bool aggregate_with(ValueT a);

	/**
	 * @brief Rebuilds the value in this RangedMaxCombiner from values of its two
	 * children's RangedMaxCombiner
	 *
	 * This sets the maximum currently stored at this combiner to the maximum of
	 * the left_child_combiner's value plus left_edge_val and
	 * right_child_combiner's value plus right_edge_val.
	 *
	 * @param my_point				       The point of the node
	 * this combiner belongs to
	 * @param left_child_combiner		 The RangedMaxCombiner of the left child
	 * of this node
	 * @param left_edge_val					 The agg_left
	 * value of this node
	 * @param right_child_combiner	 The RangedMaxCombiner of the right
	 * child of this node
	 * @param left_edge_val					 The agg_right
	 * value of this node
	 * @return FIXME ignored for now
	 */
	bool rebuild(KeyT my_point, const MyType * left_child_combiner,
	             ValueT left_edge_val, const MyType * right_child_combiner,
	             ValueT right_edge_val) noexcept(dyn_segtree_internal::
	                                                 noexcept_math<ValueT>());

	/**
	 * @brief Returns the currently stored combined value in this combiner
	 *
	 * @return the currently stored combined value in this combiner
	 */
	ValueT get() const noexcept;

	/**
	 * @brief Returns whether the maximum stored in this RangedMaxCombiner is
	 * bounded to the left
	 *
	 * If this method returns false, the value of get_left_border() is not
	 * meaningful, and the maximum stored in this combiner should be treated to
	 * extend all the way to the left.
	 *
	 * **Note**: This should never happen with combiners retrieved via
	 * get_combiner().
	 *
	 * @return See above
	 */
	bool is_left_border_valid() const noexcept;
	/**
	 * @brief Returns whether the maximum stored in this RangedMaxCombiner is
	 * bounded to the right
	 *
	 * If this method returns false, the value of get_right_border() is not
	 * meaningful, and the maximum stored in this combiner should be treated to
	 * extend all the way to the right.
	 *
	 * **Note**: This should never happen with combiners retrieved via
	 * get_combiner().
	 *
	 * @return See above
	 */
	bool is_right_border_valid() const noexcept;

	/**
	 * @brief Returns the left border of the interval over which the maximum
	 * stored in this combiner occurs.
	 *
	 * If there are multiple disjunct intervals during which the maximum value
	 * occurs, the leftmost such interval is returned.
	 *
	 * @return The left border of the maximum interval
	 */
	KeyT get_left_border() const noexcept;

	/**
	 * @brief Returns the right border of the interval over which the maximum
	 * stored in this combiner occurs.
	 *
	 * If there are multiple disjunct intervals during which the maximum value
	 * occurs, the leftmost such interval is returned.
	 *
	 * @return The right border of the maximum interval
	 */
	KeyT get_right_border() const noexcept;

	// TODO DEBUG
	static std::string
	get_name()
	{
		return "RangedMaxCombiner";
	}
	// TODO DEBUG
	std::string
	get_dbg_value() const
	{
		std::string res = std::to_string(this->val) + std::string("@[");
		if (this->left_border_valid) {
			res += std::to_string(this->left_border);
		} else {
			res += std::string("--");
		}
		res += ":";
		if (this->right_border_valid) {
			res += std::to_string(this->right_border);
		} else {
			res += "--";
		}
		res += "]";

		return res;
	}

private:
	ValueT val;

	// TODO replace by std::optional when switching to C++17
	KeyT left_border;
	bool left_border_valid;
	KeyT right_border;
	bool right_border_valid;

	ValueT child_value(const MyType * child) const noexcept;
};

/**
 * @brief This class represents the pack of combiners associated with every node
 * of a Dynamic Segment Tree
 *
 * A DynamicSegmentTree can have multiple combiners associated with each node.
 * See TODO for details. Every combiner allows to retrieve a different combined
 * metric (such as maximum, minimum, …) of the aggregate values over arbitrary
 * ranges in the Dynamic Segment Tree.
 *
 * @tparam KeyT       The type of the interval borders
 * @tparam AggValueT	The type of the aggregate values in your
 * DynamicSegmentTree
 * @tparam Combiners	A list of combiner classes
 */
template <class KeyT, class AggValueT, class... Combiners>
class CombinerPack {
public:
	using MyType = CombinerPack<KeyT, AggValueT, Combiners...>;

	/**
	 * @brief Rebuilds all combiners at this node from its children's combiners
	 *
	 * This method calls the rebuild() method on all combiners attached to this
	 * node with the respective combined values from the left / right child.
	 *
	 * @param my_point				The point of the node this
	 * CombinerPack belongs to
	 * @param left_child 			The CombinerPack of the left child (or
	 * nullptr)
	 * @param left_edge_val   The agg_left value of this node
	 * @param right_child			The CombinerPack of the right child (or
	 * nullptr)
	 * @param right_edge_val	The agg_right value of this node
	 * @return TODO IGNORED
	 */
	bool rebuild(
	    KeyT my_point, const MyType * left_child, AggValueT left_edge_val,
	    const MyType * right_child,
	    AggValueT right_edge_val) noexcept(dyn_segtree_internal::
	                                           noexcept_all_combiners<
	                                               AggValueT, Combiners...>());

	// TODO the bool is only returned for sake of expansion! Fix that!
	bool collect_left(
	    KeyT my_point, const MyType * left_child_combiner,
	    AggValueT edge_val) noexcept(dyn_segtree_internal::
	                                     noexcept_all_combiners<AggValueT,
	                                                            Combiners...>());
	bool collect_right(
	    KeyT my_point, const MyType * right_child_combiner,
	    AggValueT edge_val) noexcept(dyn_segtree_internal::
	                                     noexcept_all_combiners<AggValueT,
	                                                            Combiners...>());

	// TODO the bool is only returned for sake of expansion! Fix that!
	bool traverse_left_edge_up(KeyT new_point, AggValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_all_combiners<AggValueT, Combiners...>());

	bool traverse_right_edge_up(KeyT new_point, AggValueT edge_val) noexcept(
	    dyn_segtree_internal::noexcept_all_combiners<AggValueT, Combiners...>());

	/**
	 * @brief Returns the combined value of a combiner contained in this
	 * CombinerPack
	 *
	 * @tparam Combiner The class of the combiner that you want the combined value
	 * of
	 * @return The combined value of the combiner specified in the Combiner
	 * template parameter
	 */
	// TODO find the correct noexcept specification
	template <class Combiner>
	typename Combiner::ValueT get() const noexcept;

	/**
	 * @brief Returns a combiner contained in this CombinerPack
	 *
	 * @tparam Combiner The class of the combiner that you want
	 * @return The combiner specified in the Combiner template parameter
	 */
	template <class Combiner>
	const Combiner & get_combiner() const noexcept;

	using pack = std::tuple<Combiners...>;

private:
	template <class Combiner>
	const Combiner * child_combiner(const MyType * child) const noexcept;

	std::tuple<Combiners...> data;
};

template <class KeyT, class AggValueT>
using EmptyCombinerPack = CombinerPack<KeyT, AggValueT>;

/**
 * @brief Base class (template) to supply your node class with metainformation
 *
 * The class you use as nodes for the Dynamic Segment Tree *must* derive from
 * this class (template). It supplies your class with the necessary members to
 * contain the linking between the tree nodes.
 *
 * @tparam KeyType				The type of the key, i.e., the
 * interval borders
 * @tparam ValueType 			The type of the values that every
 * interval is associated with
 * @tparam AggValueType		The typo of an aggregate of multiple
 * ValueT_in's. See DOCTODO for details.
 * @tparam TreeSelector               Specifies which balanced binary tree
 * implementation to use for the underlying tree. Must be one of UseRBTree<...>
 * (to use the red-black tree), UseZipTree<...> (to use the zip tree) or
 * UseWBTree<...> (to use the weight-balanced tree). You need to specify the
 * same selector at the DynamicSegmentTree!
 * @tparam Tag 						The tag used to identify
 * the tree that this node should be inserted into. See RBTree for details.
 */
template <class KeyType, class ValueType, class AggValueType, class Combiners,
          class TreeSelector = UseDefaultRBTree, class Tag = int>
class DynSegTreeNodeBase {
	// TODO why is all of this public?
public:
	/// @cond INTERNAL
	using KeyT = KeyType;
	using ValueT = ValueType;
	using AggValueT = AggValueType;
	using MyClass = DynSegTreeNodeBase<KeyType, ValueType, AggValueType,
	                                   Combiners, TreeSelector, Tag>;

	using InnerNode = dyn_segtree_internal::InnerNode<
	    TreeSelector::template InnerNodeBaseBuilder<
	        typename TreeSelector::template Tag<Tag>>::template Base,
	    MyClass, KeyT, ValueT, AggValueT, Combiners, Tag>;

	// TODO make these private
	/**
	 * @brief RBTree node that represents the start of the interval represented by
	 * this DynSegTreeNodeBase
	 */
	InnerNode start;

	/**
	 * @brief RBTree node that represents the end of the interval represented by
	 * this DynSegTreeNodeBase
	 */
	InnerNode end;

	/* Methods to be used for benchmarking purposes only! */
	template <class Dummy = TreeSelector>
	std::enable_if_t<utilities::is_specialization<Dummy, UseZipTree>{}, void>
	benchmark_update_inner_ranks() noexcept
	{
		this->start.update_rank();
		this->end.update_rank();
	}

	/// @endcond
};

/**
 * @brief You must derive your own traits class from this class template,
 * telling the DynamicSegmentTree how to interact with your node class.
 *
 * You must derive from this class template and supply the DynamicSegmentTree
 * with your own derived class. At the least, you have to implement the methods
 * get_lower, get_upper and get_value for the DynamicSegmentTree to work. See
 * the respective methods' documentation for details.
 *
 * @tparam Node 	Your node class to be used in the DynamicSegmentTree,
 * derived from DynSegTreeNodeBase
 */
template <class Node> // TODO move this into the methods
class DynSegTreeNodeTraits {
public:
	/**
	 * The type of the borders of intervals / segments in the DynamicSegmentTree
	 */
	using KeyT = typename Node::KeyT;
	/**
	 * The type of the values associated with the intervals in the
	 * DynamicSegmentTree
	 */
	using ValueT = typename Node::ValueT;

	/**
	 * Must be implemented to return the lower bound of the interval represented
	 * by n.
	 *
	 * @param n The node whose lower interval bound should be returned.
	 * @return Must return the lower interval bound of n
	 */
	static KeyT get_lower(const Node & n);

	/**
	 * Must be implemented to return the upper bound of the interval represented
	 * by n.
	 *
	 * @param n The node whose upper interval bound should be returned.
	 * @return Must return the upper interval bound of n
	 */
	static KeyT get_upper(const Node & n);

	/**
	 * Should be implemented to indicate whether an interval contains its lower
	 * border or not.
	 *
	 * The default (if this method is not implemented) is true.
	 */
	static bool
	is_lower_closed(const Node & n)
	{
		(void)n;
		return true;
	};

	/**
	 * Should be implemented to indicate whether an interval contains its upper
	 * border or not.
	 *
	 * The default (if this method is not implemented) is false.
	 */
	static bool
	is_upper_closed(const Node & n)
	{
		(void)n;
		return false;
	};

	/**
	 * Must be implemented to return the value associated with the interval
	 * represented by n.
	 *
	 * @param n The node whose associated value should be returned
	 * @return Must return the value associated with n
	 */
	static ValueT get_value(const Node & n);
};

/**
 * @brief The Dynamic Segment Tree class
 *
 * This class provides a dynamic version of a segment tree. For details on the
 * implementation, see DOCTODO. The dynamic segment tree provides the following
 * operations:
 *
 * * Querying for the aggregate value ta a point x (a "stabbing query") in O(log
 * n) * A
 * * Insertion of a new interval in O(log n) * A
 * * Deletion of an interval in O(log n) * A
 *
 * where n is the number of intervals in the dynamic segment tree and A is the
 * time it takes to aggregate a value, i.e., compute operator+(AggValueT,
 * ValueT).
 *
 * The DynamicSegmentTree can be based either on a red-black tree (the RBTree),
 * or on a Zip Tree (the ZTree). By default, the red-black tree is used.
 * However, especially for applications where segments are moved frequently, the
 * Zip Tree has proven to be more efficient. You select the underlying tree via
 * the TreeSelector template parameter.
 *
 * DOCTODO combiners
 *
 * @tparam Node					The node class in your tree,
 * must be derived from DynSegTreeNodeBase
 * @tparam NodeTraits		The node traits for your node class, must be
 * derived from DynSegTreeNodeTraits
 * @tparam Options			Options for this tree. See DOCTODO for
 * details.
 * @tparam TreeSelector               Specifies which balanced binary tree
 * implementation to use for the underlying tree. Must be one of UseRBTree<...>
 * (to use the red-black tree), UseZipTree<...> (to use the zip tree) or
 * UseWBTree<...> (to use the weight-balanced tree). You need to specify the
 * same selector at the DynSegTreeNodeBase!
 * @tparam Tag					The tag of this tree. Allows to
 * insert the same node in multiple dynamic segment trees. See DOCTODO for
 * details.
 */
// TODO DOC right-open intervals

// TODO constant-time size
template <class Node, class NodeTraits, class Combiners,
          class Options = DefaultOptions, class TreeSelector = UseDefaultRBTree,
          class Tag = int>
class DynamicSegmentTree {
	// TODO add a static assert that checks that the types in all combiners are
	// right
private:
	using NB = DynSegTreeNodeBase<typename Node::KeyT, typename Node::ValueT,
	                              typename Node::AggValueT, Combiners,
	                              TreeSelector, Tag>;
	using InnerNode = typename NB::InnerNode;
	using InnerOptions =
	    typename TreeSelector::template Options<InnerNode, typename Node::KeyT>;

	static_assert(std::is_base_of<DynSegTreeNodeTraits<Node>, NodeTraits>::value,
	              "NodeTraits not properly derived from DynSegTreeNodeTraits!");
	static_assert(std::is_base_of<NB, Node>::value,
	              "Node class not properly derived from DynSegTreeNodeBase!");
	static_assert(Options::multiple,
	              "DynamicSegmentTree always allows multiple equal intervals.");

	static constexpr bool noexcept_ops =
	    dyn_segtree_internal::noexcept_dst<typename Node::ValueT, Combiners>();

public:
	using KeyT = typename Node::KeyT;
	using ValueT = typename Node::ValueT;
	using AggValueT = typename Node::AggValueT;
	using MyClass = DynamicSegmentTree<Node, NodeTraits, Combiners, Options,
	                                   TreeSelector, Tag>;

private:
	class InnerTree
	    : public TreeSelector::template BaseTree<InnerTree, Node, NodeTraits,
	                                             InnerNode, Tag> {
	public:
		using BaseTree =
		    typename TreeSelector::template BaseTree<InnerTree, Node, NodeTraits,
		                                             InnerNode, Tag>;

		using BaseTree::BaseTree;

		void modify_contour(InnerNode * left, InnerNode * right,
		                    ValueT val) noexcept(noexcept_ops);

		using Contour =
		    std::pair<std::vector<InnerNode *>, std::vector<InnerNode *>>;
		void build_lca(InnerNode * left, InnerNode * right) const
		    noexcept(noexcept_ops);

		static bool rebuild_combiners_at(InnerNode * n) noexcept(noexcept_ops);
		static void
		rebuild_combiners_recursively(InnerNode * n) noexcept(noexcept_ops);

	private:
		// Generation to be used to tag nodes during LCA search
		mutable size_t generation = 0;
		// Allocate-once buffers for the contour

		// TODO this is real bad style: public mutable members?
	public:
		/**
		 * @brief Result of a find_lca call.
		 *
		 * This vector stores the "left part" of a call to find_lca.
		 *
		 * @warning It is only valid after a call to find_lca, and before
		 * the next call to find_lca!
		 */
		mutable std::vector<InnerNode *> contour_left_path;
		/**
		 * @brief Result of a find_lca call.
		 *
		 * This vector stores the "left part" of a call to find_lca.
		 *
		 * @warning It is only valid after a call to find_lca, and before
		 * the next call to find_lca!
		 */
		mutable std::vector<InnerNode *> contour_right_path;
	};

public:
	/**
	 * @brief Move constructor
	 **/
	DynamicSegmentTree(MyClass && other) noexcept(noexcept_ops);

	/**
	 * @brief Default constructor
	 **/
	DynamicSegmentTree() noexcept(noexcept_ops);

	/**
	 * @brief Insert an interval into the dynamic segment tree
	 *
	 * This inserts the interval represented by the node n into the dynamic
	 * segment tree. The interval may not be empty.
	 *
	 * @param n		The node representing the interval being inserted
	 */
	void insert(Node & n) noexcept(noexcept_ops);

	/**
	 * @brief Removes an intervals from the dynamic segment tree
	 *
	 * Removes the (previously inserted) node n from the dynamic segment tree
	 *
	 * @param n 	The node to be removed
	 */
	void remove(Node & n) noexcept(noexcept_ops);

	/**
	 * @brief Returns whether the dynamic segment tree is empty
	 *
	 * This method runs in O(1).
	 *
	 * @return true if the dynamic segment tree is empty, false otherwise
	 */
	bool empty() const noexcept;

	/**
	 * @brief Perform a stabbing query at point x
	 *
	 * This query asks for the aggregate value over all intervals containing point
	 * x. This is a "stabbing query".
	 *
	 * @param 		x The point to query for
	 * @return 		The aggregated value for all intervals containing x
	 */
	AggValueT query(const typename Node::KeyT & x) const noexcept;

	template <class Combiner>
	Combiner get_combiner() const noexcept(noexcept_ops);

	template <class Combiner>
	Combiner get_combiner(const typename Node::KeyT & lower,
	                      const typename Node::KeyT & upper,
	                      bool lower_closed = true,
	                      bool upper_closed = false) const noexcept(noexcept_ops);

	template <class Combiner>
	typename Combiner::ValueT get_combined() const noexcept(noexcept_ops);

	template <class Combiner>
	typename Combiner::ValueT get_combined(const typename Node::KeyT & lower,
	                                       const typename Node::KeyT & upper,
	                                       bool lower_closed = true,
	                                       bool upper_closed = false) const
	    noexcept(noexcept_ops);

	/*
	 * Iteration
	 */
	template <bool reverse>
	using const_iterator = typename InnerTree::template const_iterator<reverse>;
	template <bool reverse>
	using iterator = typename InnerTree::template iterator<reverse>;

	// TODO derive a non-internal class from the InnerNode, and make the iterator
	// return a pointer to that.

	/**
	 * Returns an iterator pointing to the smallest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<false> cbegin() const noexcept;
	/**
	 * Returns an iterator pointing after the largest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<false> cend() const noexcept;
	/**
	 * Returns an iterator pointing to the smallest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<false> begin() const noexcept;
	iterator<false> begin() noexcept;

	/**
	 * Returns an iterator pointing after the largest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<false> end() const noexcept;
	iterator<false> end() noexcept;

	/**
	 * Returns an reverse iterator pointing to the largest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<true> crbegin() const noexcept;
	/**
	 * Returns an reverse iterator pointing before the smallest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<true> crend() const noexcept;
	/**
	 * Returns an reverse iterator pointing to the largest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<true> rbegin() const noexcept;
	iterator<true> rbegin() noexcept;

	/**
	 * Returns an reverse iterator pointing before the smallest \ref
	 * dyn_segtree_internal::InnerNode "InnerNode" representing a start or end
	 * event.
	 */
	const_iterator<true> rend() const noexcept;
	iterator<true> rend() noexcept;

	/**
	 * Returns an iterator to the first event the key of which is not less than
	 * <key>
	 *
	 * @param key The key to search for.
	 * @return	An iterator to the first event the key of which is not less than
	 * <key>
	 */
	const_iterator<false>
	lower_bound_event(const typename Node::KeyT & key) const noexcept;
	iterator<false> lower_bound_event(const typename Node::KeyT & key) noexcept;

	/**
	 * Returns an iterator to the first event the key of which is greater than
	 * <key>
	 *
	 * @param key The key to search for.
	 * @return	An iterator to the first event the key of which is greater than
	 * <key>
	 */
	const_iterator<false>
	upper_bound_event(const typename Node::KeyT & key) const noexcept;
	iterator<false> upper_bound_event(const typename Node::KeyT & key) noexcept;

	/**
	 * @brief Removes all elements from the tree.
	 *
	 * TODO write a test for this method
	 *
	 * Removes all elements from the tree.
	 */
	void clear() noexcept(noexcept_ops);

	/*
	 * DEBUGGING
	 */
	void dbg_verify() const;

	template <class Combiner>
	void dbg_verify_max_combiner() const;

private:
	// TODO build a generic function for this
	template <class... Ts>
	using NodeNameGetterCurried =
	    dyn_segtree_internal::ASCIIInnerNodeNameGetter<InnerNode, Ts...>;
	using NodeNameGetter =
	    typename utilities::pass_pack<typename Combiners::pack,
	                                  NodeNameGetterCurried>::type;
	template <class... Ts>
	using DotNameGetterCurried =
	    dyn_segtree_internal::DOTInnerNodeNameGetter<InnerNode, Ts...>;
	using DotNameGetter =
	    typename utilities::pass_pack<typename Combiners::pack,
	                                  DotNameGetterCurried>::type;

	using TreePrinter = debug::TreePrinter<InnerNode, NodeNameGetter>;
	using TreeDotExporter = debug::TreeDotExport<
	    InnerNode, DotNameGetter,
	    dyn_segtree_internal::DOTInnerEdgeNameGetter<InnerNode>>;

public:
	// TODO Debugging only!
	void dbg_print_inner_tree() const;
	std::stringstream & dbg_get_dot() const;

private:
	void apply_interval(Node & n) noexcept(noexcept_ops);
	void unapply_interval(Node & n) noexcept(noexcept_ops);

	InnerTree t;

	void dbg_verify_all_points() const;
	void dbg_verify_start_end() const;

#ifdef YGG_STORE_SEQUENCE_DST
	mutable typename ::ygg::utilities::BenchmarkSequenceStorage<
	    std::pair<typename Node::KeyT, typename Node::KeyT>, typename Node::KeyT,
	    typename Options::SequenceInterface::ValueT>
	    bss;
#endif
};

} // namespace ygg

#ifndef YGG_DYNAMIC_SEGMENT_TREE_CPP
#include "dynamic_segment_tree.cpp"
#endif

#endif // YGG_DYNAMIC_SEGMENT_TREE_HPP
