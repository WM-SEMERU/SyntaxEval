# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/masker.ipynb.

# %% auto 0
__all__ = ['Masker']

# %% ../nbs/masker.ipynb 2
import CodeCheckList.utils as utils
from .tokenizer import CodeTokenizer
from tree_sitter import Parser
import random

# %% ../nbs/masker.ipynb 4
class Masker():
    """Masker module"""
    def __init__(self, code_tokenizer: CodeTokenizer):
        self.code_tokenizer = code_tokenizer

    def mask_random_tokens(self,                     #self
                           encoding: list,           #list of encodings
                           number_tokens_to_mask:int #number of tokens to mask
                           ):
        node_mask_id = self.code_tokenizer.tokenizer.mask_token_id
        positions = random.sample(range(len(encoding['input_ids'])), number_tokens_to_mask)
        for idx in positions:
            encoding['input_ids'][idx]=node_mask_id
        return encoding

    def mask_ast_tokens(
    self,                            #self
    code: str,                       #source code snippet to mask
    encoding: list,                  #list of encodings
    target_node_type_id: int,        #target node type id to search, -1: random masking
    masking_rate: float,             #masking rate to apply [0-1]
    ):  
        node_mask_id = self.code_tokenizer.tokenizer.mask_token_id
        tree = self.code_tokenizer.parser.parse(bytes(code, "utf8"))
        filtered_nodes = []
        utils.find_nodes(tree.root_node, self.code_tokenizer.node_types[target_node_type_id], filtered_nodes)
        filtered_node_offsets = [(utils.convert_to_offset(node.start_point, code.split("\n")), 
            utils.convert_to_offset(node.end_point, code.split("\n"))) for node in filtered_nodes]
        filtered_node_offsets = utils.get_elements_by_percentage(filtered_node_offsets, masking_rate)
        number_of_masked_tokens = 0
        for index, offset in enumerate(encoding['offset_mapping']):
            for filtered_node_offset in filtered_node_offsets:
                if (filtered_node_offset[0] <= offset[0] and offset[0] < filtered_node_offset[1]) \
                or (filtered_node_offset[0] < offset[1] and offset[1] <= filtered_node_offset[1]) \
                or (filtered_node_offset[0] >= offset[0] and offset[1] >= filtered_node_offset[1]) :
                #if offset[0] >= filtered_node_offset[0] and offset[1]<= filtered_node_offset[1]:
                    encoding['input_ids'][index] = node_mask_id
                    number_of_masked_tokens += 1
        return encoding, number_of_masked_tokens
