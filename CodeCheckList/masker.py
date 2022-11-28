# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/masker.ipynb.

# %% auto 0
__all__ = ['find_nodes', 'convert_to_offset', 'Masker']

# %% ../nbs/masker.ipynb 2
import CodeCheckList
from .tokenizer import CodeTokenizer
from tree_sitter import Parser

# %% ../nbs/masker.ipynb 3
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

# %% ../nbs/masker.ipynb 4
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

# %% ../nbs/masker.ipynb 5
class Masker():
    """Masker module"""
    def __init__(self, code_tokenizer: CodeTokenizer):
        self.code_tokenizer = code_tokenizer

    def __call__(
    self,                            #self
    code: str,                       #source code snippet to mask
    encoding: list,                  #list of encodings
    target_node_type_id: int         #target node type id to search 
    ):  
        node_mask_id = self.code_tokenizer.tokenizer.mask_token_id
        tree = self.code_tokenizer.parser.parse(bytes(code, "utf8"))

        filtered_nodes = []
        find_nodes(tree.root_node, self.code_tokenizer.node_types[target_node_type_id], filtered_nodes)
        filtered_node_offsets = [(convert_to_offset(node.start_point, code.split("\n")), 
            convert_to_offset(node.end_point, code.split("\n"))) for node in filtered_nodes]

        for index, offset in enumerate(encoding['offset_mapping']):
            for filtered_node_offset in filtered_node_offsets:
                if offset[0] >= filtered_node_offset[0] and offset[1]<= filtered_node_offset[1]:
                    encoding['input_ids'][index] = node_mask_id
        return encoding
