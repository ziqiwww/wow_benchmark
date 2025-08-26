#ifndef INTERVALTREE_HPP
#define INTERVALTREE_HPP

#include "rbtree.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace ygg {
namespace intervaltree_internal {
template <class Node, class INB, class NodeTraits, bool skipfirst,
          class Comparable>
Node * find_next_overlapping(Node * cur, const Comparable & q);

template <class KeyType>
class DummyRange : public std::pair<KeyType, KeyType> {
public:
	DummyRange(KeyType lower, KeyType upper);
};

template <class Node, class NodeTraits, bool sort_upper>
class IntervalCompare {
public:
	template <class T1, class T2>
	bool operator()(const T1 & lhs, const T2 & rhs) const;
};

// TODO add a possibility for bulk updates
template <class Node, class INB, class NodeTraits>
class ExtendedNodeTraits : public NodeTraits {
public:
	// TODO these can probably be made more efficient
	template <class BaseTree>
	static void leaf_inserted(Node & node, BaseTree & t);

	static void fix_node(Node & node);
	template <class BaseTree>
	static void rotated_left(Node & node, BaseTree & t);
	template <class BaseTree>
	static void rotated_right(Node & node, BaseTree & t);
	template <class BaseTree>
	static void deleted_below(Node & node, BaseTree & t);
	template <class BaseTree>
	static void
	delete_leaf(Node & node, BaseTree & t)
	{
		(void)node;
		(void)t;
	}
	template <class BaseTree>
	static void swapped(Node & n1, Node & n2, BaseTree & t);

	// Make our DummyRange comparable
	static typename NodeTraits::key_type get_lower(
	    const intervaltree_internal::DummyRange<typename NodeTraits::key_type> &
	        range);
	static typename NodeTraits::key_type get_upper(
	    const intervaltree_internal::DummyRange<typename NodeTraits::key_type> &
	        range);
};
} // namespace intervaltree_internal

template <class Node, class NodeTraits, class Options = DefaultOptions,
          class Tag = int>
class ITreeNodeBase : public RBTreeNodeBase<Node, Options, Tag> {
public:
	typename NodeTraits::key_type _it_max_upper;
};

/**
 * @brief Abstract base class for the Node Traits that need to be implemented
 *
 * Every Interval Tree needs to be supplied with a node traits class that must
 * be derived from this class. In your derived class, you must define the
 * key_type as the type of your interval's bounds, and you must implement
 * get_lower() and get_upper() to return the interval bounds of your nodes.
 */
template <class Node>
class ITreeNodeTraits {
public:
	/**
	 * @brief The type of your interval bounds. This is the type that get_lower()
	 * and get_upper() must return. This type must be comparable, i.e., operator<
	 * etc. must be implemented.
	 */
	using key_type = void;

	/**
	 * Must be implemented to return the lower bound of the interval represented
	 * by n.
	 *
	 * @param n The node whose lower interval bound should be returned.
	 * @return Must return the lower interval bound of n
	 */
	static key_type get_lower(const Node & n) = delete;

	/**
	 * Must be implemented to return the upper bound of the interval represented
	 * by n.
	 *
	 * @param n The node whose upper interval bound should be returned.
	 * @return Must return the upper interval bound of n
	 */
	static key_type get_upper(const Node & n) = delete;
};

/**
 * @brief Stores an Interval Tree
 *
 * This class stores an interval tree on the nodes it contains. It is
 * implemented via the 'augmented red-black tree' described by Cormen et al.
 *
 * @tparam Node 				The node class for this Interval Tree. Must
 * be derived from ITreeNodeBase.
 * @tparam NodeTraits 	The node traits for this Interval Tree. Must be derived
 * from
 * @tparam Options			Passed through to RBTree. See there for
 * documentation.
 * @tparam Tag					Used to add nodes to multiple interval
 * trees. See RBTree documentation for details.
 */
template <class Node, class NodeTraits, class Options = DefaultOptions,
          class Tag = int>
class IntervalTree
    : private RBTree<
          Node,
          intervaltree_internal::ExtendedNodeTraits<
              Node, ITreeNodeBase<Node, NodeTraits, Options, Tag>, NodeTraits>,
          Options, Tag,
          intervaltree_internal::IntervalCompare<Node, NodeTraits,
                                                 Options::itree_fast_find>> {
public:
	using Key = typename NodeTraits::key_type;
	using MyClass = IntervalTree<Node, NodeTraits, Options, Tag>;

	using INB = ITreeNodeBase<Node, NodeTraits, Options, Tag>;
	static_assert(std::is_base_of<INB, Node>::value,
	              "Node class not properly derived from ITreeNodeBase!");

	static_assert(std::is_base_of<ITreeNodeTraits<Node>, NodeTraits>::value,
	              "NodeTraits not properly derived from ITreeNodeTraits!");

	using ENodeTraits =
	    intervaltree_internal::ExtendedNodeTraits<Node, INB, NodeTraits>;
	using BaseTree =
	    RBTree<Node,
	           intervaltree_internal::ExtendedNodeTraits<Node, INB, NodeTraits>,
	           Options, Tag,
	           intervaltree_internal::IntervalCompare<Node, NodeTraits,
	                                                  Options::itree_fast_find>>;

	IntervalTree();

	bool verify_integrity() const;
	void dump_to_dot(const std::string & filename) const;

	/* Import some of RBTree's methods into the public namespace */
	using BaseTree::empty;
	using BaseTree::insert;
	using BaseTree::remove;

	// Iteration of sets of intervals
	template <class Comparable>
	class QueryResult {
	public:
		class const_iterator {
		public:
			typedef ptrdiff_t difference_type;
			typedef Node value_type;
			typedef const Node & const_reference;
			typedef const Node * const_pointer;
			typedef std::input_iterator_tag iterator_category;

			const_iterator(Node * n, const Comparable & q);
			const_iterator(const const_iterator & other);
			~const_iterator();

			const_iterator & operator=(const const_iterator & other);

			bool operator==(const const_iterator & other) const;
			bool operator!=(const const_iterator & other) const;

			const_iterator & operator++();
			const_iterator operator++(int);

			const_reference operator*() const;
			const_pointer operator->() const;

		private:
			Node * n;
			Comparable q;
		};

		QueryResult(Node * n, const Comparable & q);

		const_iterator begin() const;
		const_iterator end() const;

	private:
		Node * n;
		Comparable q;
	};

	/**
	 * @brief Queries intervals contained in the interval tree
	 *
	 * This method queries for intervals overlapping a query interval.
	 * The query parameter q can be anything that is comparable to an interval.
	 * A class <Comparable> is comparable to an interval if the NodeTraits
	 * implement a get_lower(const Comparable &) and get_upper(const Comparable &)
	 * method.
	 *
	 * The return value is a QueryResult that contains all intervals that overlap
	 * the given query.
	 *
	 * @param q Anything that is comparable (i.e., has get_lower() and get_upper()
	 * methods in NodeTraits) to an interval
	 * @result A QueryResult holding all intervals in the tree that overlap q
	 */
	template <class Comparable>
	QueryResult<Comparable> query(const Comparable & q) const;

	/**
	 * @brief Checks if a specified interval is contained in the interval tree
	 *
	 * This method queries whether a specified interval is contained in the tree.
	 * Note that it only returns an interval that has the *exact* same borders as
	 * the supplied query. To query for overlap, see the query method.
	 *
	 * The return value is an iterator to a found interval in the tree, or end()
	 * if no such interval is in the tree.
	 *
	 * This method runs in O(log n) if the ITREE_FAST_FIND option is set,
	 * and in O(log n + k) otherwise (with k being the number of intervals
	 * overlapping the query).
	 *
	 * @param q Anything that is comparable (i.e., has get_lower() and get_upper()
	 * methods in NodeTraits) to an interval
	 * @result An iterator pointing to the requested interval, or end() if no such
	 *  interval exists.
	 */
	template <class Comparable>
	typename BaseTree::template const_iterator<false>
	find(const Comparable & q) const;
	template <class Comparable>
	typename BaseTree::template iterator<false> find(const Comparable & q);

	template <class Comparable>
	typename BaseTree::template const_iterator<false>
	interval_upper_bound(const Comparable & query_range) const;

	// TODO FIXME this is actually very specific?
	void fixup_maxima(Node & lowest);

	// Iterating the events should still be possible
	using BaseTree::begin;
	using BaseTree::end;

private:
	bool verify_maxima(Node * n) const;

	template <class Comparable>
	typename BaseTree::template iterator<false> find_slow(const Comparable & q);

	template <class Comparable>
	typename BaseTree::template iterator<false> find_fast(const Comparable & q);
};

} // namespace ygg

#include "intervaltree.cpp"

#endif // INTERVALTREE_HPP
