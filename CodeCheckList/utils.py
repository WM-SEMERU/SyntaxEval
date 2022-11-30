# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['traverse', 'find_nodes', 'unroll_node_types', 'convert_to_offset', 'get_sub_set_test_set']

# %% ../nbs/utils.ipynb 2
import CodeCheckList

# %% ../nbs/utils.ipynb 3
# From: https://github.com/github/CodeSearchNet/tree/master/function_parser
def traverse(
    node,       # tree-sitter node
    results,    # list to append results to
) -> None:
    """Traverse in a recursive way, a tree-sitter node and append results to a list."""
    if node.type == 'string':
        results.append(node)
        return
    for n in node.children:
        traverse(n, results)
    if not node.children:
        results.append(node)

# %% ../nbs/utils.ipynb 4
def find_nodes(
    node,               #Tree sitter ast treee
    target_node_type,   #Target node type to search in the tree
    results,            #List to append the resutls to
) -> None: 
    """Traverses the tree and find the specified node type"""
    if node.type == target_node_type:
        results.append(node)
        return
    for n in node.children:
        find_nodes(n, target_node_type, results)

# %% ../nbs/utils.ipynb 5
def unroll_node_types(
    nested_node_types: dict, # node_types from tree-sitter
) -> list: # list of node types
    """Unroll nested node types into a flat list of node types. This includes subtypes as well."""
    node_types = [node_type["type"] for node_type in nested_node_types]
    node_subtypes = [
        node_subtype["type"]
        for node_type in node_types
        if "subtypes" in node_type
        for node_subtype in node_type["subtypes"]
    ]
    return list(set(node_types + node_subtypes))

# %% ../nbs/utils.ipynb 6
def convert_to_offset(
    point,              #point to convert
    lines: list         #list of lines in the source code
    ):
        """Convert the point to an offset"""
        row, column = point
        chars_in_rows = sum(map(len, lines[:row])) + row
        chars_in_columns = len(lines[row][:column])
        offset = chars_in_rows + chars_in_columns
        return offset

# %% ../nbs/utils.ipynb 7
def get_sub_set_test_set(test_set, test_size:int):
    sub_samples = []
    for sample in test_set:
        sub_samples.append(sample)
        if len(sub_samples)>=test_size:
            break
    return sub_samples
