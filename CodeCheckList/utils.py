# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['traverse', 'find_nodes', 'get_node_type_list', 'unroll_node_types', 'convert_to_offset', 'get_sub_set_test_set',
           'get_random_sub_set_test_set', 'get_test_sets', 'get_test_sets_galeras', 'get_elements_by_percentage']

# %% ../nbs/utils.ipynb 2
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
def get_node_type_list(
    node
) -> None:
    """Traverses the tree and get all the node types"""
    node_types = []
    def traverse_and_get_types(node, node_type_set):
        node_type_set.append(node.type)
        for n in node.children:
            traverse_and_get_types(n, node_type_set)
    traverse_and_get_types(node, node_types)
    return node_types

# %% ../nbs/utils.ipynb 7
def unroll_node_types(
    nested_node_types: dict  # node_types from tree-sitter
) -> list: # list of node types
    def iterate_and_unroll_dict(nested_node_types: dict, all_node_types: set):
        for key, value in nested_node_types.items():
            if key == 'type' and type(value) == str:
                all_node_types.add(value)
            if type(value) == dict:
                iterate_and_unroll_dict(value, all_node_types)
            if type(value) == list:
                for element in value:
                    iterate_and_unroll_dict(element, all_node_types) 
    all_node_types = set()
    for dictionary in nested_node_types:
        iterate_and_unroll_dict(dictionary, all_node_types)
    all_node_types.add('ERROR')
    return list(all_node_types)

# %% ../nbs/utils.ipynb 8
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

# %% ../nbs/utils.ipynb 9
def get_sub_set_test_set(test_set, test_size:int):
    sub_samples = []
    for sample in test_set:
        sub_samples.append(sample)
        if len(sub_samples)>=test_size:
            break
    return sub_samples

# %% ../nbs/utils.ipynb 10
def get_random_sub_set_test_set(test_set, test_size:int):
    sub_samples = []
    while len(sub_samples)<test_size:
        random_index = random.randrange(0,len(test_set))
        sub_samples.append(test_set[random_index])
    return sub_samples

# %% ../nbs/utils.ipynb 11
def get_test_sets(test_set, language, max_token_number, model_tokenizer, with_ranks=False, num_proc=1):
    subset = test_set.filter(lambda sample: True if sample['language']== language 
            and len(sample['func_code_tokens']) < max_token_number
            and len(model_tokenizer.tokenizer(sample['whole_func_string'])['input_ids']) < max_token_number
            else False, num_proc=num_proc)
    return subset

# %% ../nbs/utils.ipynb 12
def get_test_sets_galeras(test_set: list, language, max_token_number, model_tokenizer):
    values = []
    for sample in test_set:
        try: 
            if sample['language'] == language and len(model_tokenizer.tokenizer(sample['code'])['input_ids']) < max_token_number:
                values.append(sample)
        except:
            print('--------------------------ERROR-----------------------------')
            print(sample)
    return values
    #return [sample for dic in test_set for sample in dic.values() if sample['language'] == language and len(model_tokenizer.tokenizer(sample['code'])['input_ids']) < max_token_number ]

# %% ../nbs/utils.ipynb 13
def get_elements_by_percentage(elements, percentage):
    indexes = set(random.sample(list(range(len(elements))), int(percentage*len(elements))))
    return [n for i,n in enumerate(elements) if i in indexes]
