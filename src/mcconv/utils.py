from jax.tree_util import tree_flatten
import jax.numpy as jnp

def check_same_type(lst):
    if not lst:
        return True  # empty list is considered as having the same type
    first_type = type(lst[0])
    return all(type(item) == first_type for item in lst[1:])

# from jax.lib import pytree
def concatenate_trees(trees, axis=0):
    """
    Takes a list of trees and concatenates every corresponding leaf (assuming is array)
    Shallow concatenation, not recursive
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.concatenate(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)