# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/judge.ipynb.

# %% auto 0
__all__ = ['Judge']

# %% ../nbs/judge.ipynb 2
import time
from .tokenizer import CodeTokenizer
import CodeCheckList.utils as utils
from func_timeout import func_set_timeout, FunctionTimedOut
from multiprocessing import Process, Queue
import textdistance

# %% ../nbs/judge.ipynb 3
class Judge:
    """Judge Module to perform all similarity evaluations in a sandbox"""
    def __init__(self, tokenizer: CodeTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, original_source_code, predicted_source_code):
        def parse_code(queue, parser, original_source_code, predicted_source_code):
            source_code_tree = self.tokenizer.parser.parse(bytes(original_source_code, "utf8"))
            predicted_code_tree = self.tokenizer.parser.parse(bytes(predicted_source_code, "utf8"))
            predicted_code_types = utils.get_node_type_list(predicted_code_tree.root_node)
            source_code_types = utils.get_node_type_list(source_code_tree.root_node)
            jaccard_similarity = textdistance.jaccard.normalized_similarity(predicted_code_types,source_code_types)
            sorensen_dice_similarity = textdistance.sorensen_dice.normalized_similarity(predicted_code_types, source_code_types)
            levenshtein_similarity = textdistance.levenshtein.normalized_similarity(predicted_code_types,source_code_types)
            queue.put([jaccard_similarity, sorensen_dice_similarity, levenshtein_similarity])
    
        @func_set_timeout(5)
        def get_parser_result(queue):
            return queue.get()
    
        queue = Queue()
        parser_process = Process(target=parse_code, args=(queue, self.tokenizer.parser, original_source_code, predicted_source_code))
        parser_process.start()
        result = [0,0,0]
        try:
            result = get_parser_result(queue)
        except FunctionTimedOut as e:
            if parser_process.is_alive():
                print('-judge deadlock-')
                parser_process.kill()
        return result