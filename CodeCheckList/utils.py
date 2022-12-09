# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['traverse', 'find_nodes', 'get_node_type_set', 'calculate_jaccard_distance', 'calculate_sorensen_dice_distance',
           'unroll_node_types', 'convert_to_offset', 'get_sub_set_test_set', 'get_random_sub_set_test_set',
           'is_valid_code', 'get_test_sets', 'is_balanced_snippet']

# %% ../nbs/utils.ipynb 2
import CodeCheckList
import ast
import random

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
def get_node_type_set(
    node
) -> None:
    """Traverses the tree and get all the node types"""
    node_types = set()
    def traverse_and_get_types(node, node_type_set):
        node_type_set.add(node.type)
        for n in node.children:
            traverse_and_get_types(n, node_type_set)
    traverse_and_get_types(node, node_types)
    return node_types

# %% ../nbs/utils.ipynb 6
def calculate_jaccard_distance(predicted_code_types, source_code_types):
    """Calculates jaccard similarity score between the node types of each tree"""
    intersection = source_code_types.intersection(source_code_types)
    union = predicted_code_types.union(source_code_types)
    return (len(union)-len(intersection))/len(union)

# %% ../nbs/utils.ipynb 7
def calculate_sorensen_dice_distance(predicted_code_types, source_code_types):
    """Calculates SOrensen-Dice similarity score between the node types of each tree"""
    intersection = source_code_types.intersection(source_code_types)
    return 2*len(intersection)/(len(predicted_code_types)+len(source_code_types))

# %% ../nbs/utils.ipynb 9
def unroll_node_types(
    nested_node_types: dict, # node_types from tree-sitter
) -> list: # list of node types
    """Unroll nested node types into a flat list of node types. This includes subtypes as well."""
    node_types = [node_type["type"] for node_type in nested_node_types]
    node_subtypes = [
        node_subtype["type"]
        for node_type in nested_node_types
        if "subtypes" in node_type
        for node_subtype in node_type["subtypes"]
    ]
    children_subtypes = [
        children_type["type"]
        for node_type in nested_node_types
        if "children" in node_type
        for children_type in node_type["children"]["types"]
    ]
    alias_subtypes = [
        children_type["type"]
        for node_type in nested_node_types
        if "fields" in node_type and "alias" in node_type["fields"] 
        for children_type in node_type["fields"]["alias"]["types"]
    ]
    return list(set(node_types + node_subtypes + children_subtypes + alias_subtypes + ['ERROR']))

# %% ../nbs/utils.ipynb 10
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

# %% ../nbs/utils.ipynb 11
def get_sub_set_test_set(test_set, test_size:int):
    sub_samples = []
    for sample in test_set:
        sub_samples.append(sample)
        if len(sub_samples)>=test_size:
            break
    return sub_samples

# %% ../nbs/utils.ipynb 12
def get_random_sub_set_test_set(test_set, test_size:int):
    sub_samples = []
    while len(sub_samples)<test_size:
        random_index = random.randrange(0,len(test_set))
        sub_samples.append(test_set[random_index])
    return sub_samples

# %% ../nbs/utils.ipynb 13
def is_valid_code(code):
    try:
        ast.parse(code)
    except:
        return False
    return True
    

# %% ../nbs/utils.ipynb 14
def get_test_sets(test_set, language, max_token_number, model_tokenizer, with_ranks=False, num_proc=1):
    subset = test_set.filter(lambda sample: True if sample['language']== language 
            and len(sample['func_code_tokens']) < max_token_number
            and len(model_tokenizer.tokenizer(sample['whole_func_string'])['input_ids']) < max_token_number
            else False, num_proc=num_proc)
    return subset

# %% ../nbs/utils.ipynb 15
def is_balanced_snippet(snippet):
    percentage = 5
    return (len(set(snippet))/len(snippet))*100 > percentage
