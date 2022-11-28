# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/tokenizer.ipynb.

# %% auto 0
__all__ = ['unroll_node_types', 'traverse', 'get_token_type', 'CodeTokenizer']

# %% ../nbs/tokenizer.ipynb 2
import CodeCheckList
import json

from transformers import AutoTokenizer
from tree_sitter import Language, Parser

# %% ../nbs/tokenizer.ipynb 4
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

# %% ../nbs/tokenizer.ipynb 5
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

# %% ../nbs/tokenizer.ipynb 6
def get_token_type(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    nodes: list,     # list of tree-sitter nodes
    lines: list,     # list of lines in the code
) -> tuple: # (parent_type, token_type) of the token
    """Get the parent AST type and token AST type of a token."""
    def get_node_span(node):
        def convert_to_offset(point):
            row, column = point
            chars_in_rows = sum(map(len, lines[:row])) + row
            chars_in_columns = len(lines[row][:column])

            offset = chars_in_rows + chars_in_columns
            return offset
        start_span = convert_to_offset(node.start_point)
        end_span = convert_to_offset(node.end_point)
        return start_span, end_span
    
    node_spans = [get_node_span(node) for node in nodes]
    for i, span in enumerate(node_spans):
        if (span[0] <= tok_span[0] and tok_span[0] < span[1]) or (span[0] < tok_span[1] and tok_span[1] <= span[1]):
            return nodes[i].parent.type, nodes[i].type

# %% ../nbs/tokenizer.ipynb 7
class CodeTokenizer():
    """A tokenizer for code, which aligns the tokens with the AST nodes."""
    def __init__(self, tokenizer, parser, node_types):
        self.tokenizer = tokenizer
        self.parser = parser
        self.node_types = node_types
    
    def __call__(self, code):
        encoding = self.tokenizer(code, return_offsets_mapping=True)
        tree = self.parser.parse(bytes(code, "utf8"))
        nodes = []
        traverse(tree.root_node, nodes)

        encoding["ast_ids"] = []
        encoding["parent_ast_ids"] = []
        for i, (start, end) in enumerate(encoding.offset_mapping):
            if encoding["input_ids"][i] in self.tokenizer.all_special_ids:
                encoding["ast_ids"].append(-1)
                encoding["parent_ast_ids"].append(-1)
                continue
            if start == None or end == None:
                encoding["ast_ids"].append(-1)
                encoding["parent_ast_ids"].append(-1)
                continue
            type_info = get_token_type((start, end), nodes, code.split("\n"))
            if type_info is None:
                encoding["ast_ids"].append(-1)
                encoding["parent_ast_ids"].append(-1)
            else:
                parent_node_type, node_type = type_info
                try:
                    encoding["ast_ids"].append(self.node_types.index(node_type))
                    encoding["parent_ast_ids"].append(self.node_types.index(parent_node_type))
                except Exception as e:
                    print(type_info)
                    print(code)
                    print(self.tokenizer.decode(encoding["input_ids"][i]))
                    encoding["ast_ids"].append(-1)
                    encoding["parent_ast_ids"].append(-1)
                    raise e
            
        return encoding

    @staticmethod
    def from_pretrained(
        name_or_path: str,  # name or path of the tokenizer
        lang: str,          # language of the tokenizer
    ):                      # CodeTokenizer for the given language
        """Create a CodeTokenizer from a pretrained tokenizer for a given language."""
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)

        # Grab the node types from the tree-sitter language
        language = Language(f"{CodeCheckList.__path__[0]}/grammars/tree-sitter-languages.so", lang)
        node_path = f"{CodeCheckList.__path__[0]}/grammars/tree-sitter-{lang}/src/node-types.json"
        with open(node_path) as f:
            node_types = json.load(f)
        node_types = unroll_node_types(node_types)

        # Create a parser for the language
        parser = Parser()
        parser.set_language(language)
        
        return CodeTokenizer(tokenizer, parser, node_types)
