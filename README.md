CodeCheckList
================

This is the repository that host the source code of **SyntaxEval** and results of our paper 'Which Syntactic Capabilities Are Statistically Learned by Masked Language Models for Code?'. 

Our work discusses the limitations of evaluating Masked Language Models (MLM) in code completion tasks. We highlight that relying on accuracy-based measurements may lead to an overestimation of models' capabilities by neglecting the syntax rules of Programming Languages. To address these issues, we introduce a technique called **SyntaxEval** in which **Syntactic Capabilities** are used to enhance the evaluation of MLMs. SyntaxEval automates the process of masking elements in the model input based on their Abstract Syntax Trees (ASTs). We conducted a case study on two popular MLMs using data from GitHub repositories. Our results showed negative causal effects between the node types and MLMs' accuracy. We conclude that MLMs under study fails to predict some syntactic capabilities.

## Prerequisites

If you want to use GPU to perform the predictions, please make sure to
have pytorch correctly installed and GPU available to use.
https://pytorch.org

``` python
!nvidia-smi
```

    Sat Dec 17 12:50:47 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 510.108.03   Driver Version: 510.108.03   CUDA Version: 11.6     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA A40          Off  | 00000000:01:00.0 Off |                    0 |
    |  0%   69C    P0   295W / 300W |  29143MiB / 46068MiB |     82%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  NVIDIA A40          Off  | 00000000:25:00.0 Off |                    0 |
    |  0%   30C    P8    33W / 300W |     26MiB / 46068MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  NVIDIA A40          Off  | 00000000:41:00.0 Off |                    0 |
    |  0%   30C    P8    35W / 300W |     26MiB / 46068MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   3  NVIDIA A40          Off  | 00000000:61:00.0 Off |                    0 |
    |  0%   35C    P0    82W / 300W |     26MiB / 46068MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   4  NVIDIA A40          Off  | 00000000:81:00.0 Off |                    0 |
    |  0%   28C    P8    32W / 300W |     26MiB / 46068MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   5  NVIDIA A40          Off  | 00000000:A1:00.0 Off |                    0 |
    |  0%   58C    P0    90W / 300W |  42659MiB / 46068MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   6  NVIDIA A40          Off  | 00000000:C1:00.0 Off |                    0 |
    |  0%   62C    P0   296W / 300W |  42659MiB / 46068MiB |    100%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   7  NVIDIA A40          Off  | 00000000:E1:00.0 Off |                    0 |
    |  0%   67C    P0   321W / 300W |  42659MiB / 46068MiB |    100%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    0   N/A  N/A   1807667      C   python3.8                       29117MiB |
    |    1   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    2   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    3   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    4   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    5   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    5   N/A  N/A   2970647      C   julia                           42633MiB |
    |    6   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    6   N/A  N/A   1778799      C   julia                           42633MiB |
    |    7   N/A  N/A      3048      G   /usr/lib/xorg/Xorg                 23MiB |
    |    7   N/A  N/A   1783425      C   julia                           42633MiB |
    +-----------------------------------------------------------------------------+

## Installation

Create a virtual environment, you can use conda, mamba or virtualenv.
Then activate the envorinment and go to the project base path.

``` sh
mamba create code-check-list
```

``` sh
cd CodeCheckList
```

Now, install CodeCheckList using the package manager

``` sh
pip install CodeCheckList
```

## General Instructions

Each module in CodeCheckList can be used independently, you can go to
./nbs if you want to look at more detail examples for each module.:

| Module              | Purpose                                                                                                                                                                                          |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Checklist.Loader    | Download and install Tree-sitter grammars                                                                                                                                                        |
| Checklist.Tokenizer | Tokenize, Encode and Associate AST types, for he requested source code snippet using the model’s BPE and Tree-Sitter Parser                                                                      |
| Checklist.Masker    | Mask the given source code snippet on occurrences for the requested AST element with a masking rate                                                                                              |
| Checklist.Predictor | Attempts to predict the masked elements of a source code snippet using the selected model. Reports the results of top-k predictions                                                              |
| Checklist.Judge     | Compare the AST representations of original snippet and prediction to calculate similarity scores                                                                                                |
| Checklist.Evaluator | Iterates over the specified number of samples from code-search-net, mask the ast elements defined by the programming language grammar in all the snippets, and report the results in a dataframe |

## Full Evaluation pipeline

### Donwloading the grammar

First, download the grammar of the programming language of interest
using the loader module

``` python
from CodeCheckList import loader

python_language = "python"

################ LOAD GRAMMAR
languages = [python_language]
loader.download_grammars(languages)
```

    /home/svelascodimate/Documents/SEMERU/CodeCheckList/CodeCheckList/grammars

### Defining the Evaluator

Define the evaluator Component to perform the evaluation of Linguistic
Capabilities

You need to setup first some parameters

``` python
#chechpoint to use
checkpoint = "huggingface/CodeBERTa-small-v1"
#number of sample sto evaluate
number_of_samples = 5
#masking rate to apply
masking_rate = 25/100
#top-k prediction per code sample
number_of_predictions_per_sample = 3
#if GPU:3 is available, set it to True, else False
gpu_available = True
#Save Path for the dataframe results
save_path = "output/CodeBERTa-small-v1/"
```

Now, Instantiate the evaluator

``` python
from CodeCheckList.evaluator import Evaluator
evaluator = Evaluator(checkpoint, python_language, gpu_available)
```

    /home/svelascodimate/miniconda3/envs/code-check-list/lib/python3.10/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

    ------------------Loading Model into GPU------------------

### Loading the samples

Next, you need to define the source code samples to be used in the
evaluation.

``` python
from datasets import load_dataset 
import CodeCheckList.utils as utils

test_set = utils.get_test_sets(load_dataset("code_search_net", split='test'), python_language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer)
test_set = utils.get_random_sub_set_test_set(test_set, number_of_samples)
```

    No config specified, defaulting to: code_search_net/all
    Found cached dataset code_search_net (/home/svelascodimate/.cache/huggingface/datasets/code_search_net/all/1.0.0/80a244ab541c6b2125350b764dc5c2b715f65f00de7a56107a28915fac173a27)
    Parameter 'function'=<function get_test_sets.<locals>.<lambda> at 0x7f6604bcb520> of the transform datasets.arrow_dataset.Dataset.filter@2.0.1 couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
      0%|          | 0/101 [00:00<?, ?ba/s]Token indices sequence length is longer than the specified maximum sequence length for this model (517 > 512). Running this sequence through the model will result in indexing errors
    100%|██████████| 101/101 [00:17<00:00,  5.71ba/s]

### Performing the Evaluation

Now, call the evaluator to perform the evaluation of Linguistic
Capabilities

Evaluation will be conducted on each sample for each AST node type
define din the grammar of the programming language.:

``` python
print(evaluator.tokenizer.node_types)
```

    ['false', 'format_specifier', 'argument_list', 'augmented_assignment', 'exec_statement', 'true', 'exec', 'global', 'for_statement', 'for', '<<', '-=', 'module', '==', 'print', '//=', '[', 'else_clause', 'type', 'subscript', 'tuple_pattern', '<', 'match_statement', 'not_operator', '"', 'float', 'dotted_name', 'or', 'finally', 'pair', 'try_statement', '/', 'set', 'concatenated_string', 'nonlocal', 'async', 'typed_parameter', 'wildcard_import', '>=', 'expression', 'yield', 'assignment', ')', '//', 'global_statement', 'class', '+', 'import_from_statement', 'not', 'parameters', '>>=', 'case_pattern', '^=', 'set_comprehension', '_simple_statement', '*=', 'relative_import', 'as_pattern', 'del', '}', 'conditional_expression', 'pass_statement', 'and', 'as', 'escape_sequence', 'chevron', 'pattern', 'future_import_statement', 'import_prefix', 'continue_statement', 'expression_list', 'list_splat_pattern', 'except_clause', 'if_clause', 'positional_separator', 'comparison_operator', 'return_statement', ':', '(', ',', 'typed_default_parameter', ']', '_compound_statement', 'list_splat', 'named_expression', 'parenthesized_expression', '+=', 'with', 'nonlocal_statement', 'case', 'ERROR', '<>', '|=', 'unary_operator', 'list_pattern', 'ellipsis', ':=', 'list', 'assert_statement', 'function_definition', 'continue', 'else', 'default_parameter', 'delete_statement', 'list_comprehension', 'dictionary', 'identifier', 'as_pattern_target', 'decorated_definition', 'comment', '__future__', 'def', '}}', 'aliased_import', 'match', '**=', '!=', 'class_definition', 'return', 'type_conversion', '{{', '.', '<=', 'generator_expression', '>', 'keyword_argument', 'import', 'from', '|', 'block', '<<=', 'case_clause', 'elif_clause', 'string', 'expression_statement', '@', 'for_in_clause', 'interpolation', '&=', '^', 'format_expression', '-', 'decorator', 'with_item', 'primary_expression', 'finally_clause', 'print_statement', 'if_statement', '>>', 'await', 'boolean_operator', 'binary_operator', 'raise_statement', 'try', '%=', 'keyword_separator', 'import_statement', 'parenthesized_list_splat', 'with_statement', 'with_clause', '**', '@=', '%', 'break_statement', 'dictionary_comprehension', 'slice', 'assert', 'break', '~', 'pass', 'dictionary_splat', 'none', 'in', 'attribute', 'call', 'lambda_parameters', 'elif', 'integer', 'dictionary_splat_pattern', ';', '*', 'tuple', '{', 'pattern_list', '/=', '->', 'raise', 'while_statement', 'parameter', '=', 'except', 'is', 'lambda', '&', 'if', 'while']

To perform the evaluation, simply call the evaluator with the defined
parameters

``` python
results_dataframe = evaluator(test_set, number_of_predictions_per_sample, masking_rate)
```

    -------- evaluating sample:0 --------
    -------- evaluating sample:1 --------
    -------- evaluating sample:2 --------
    -------- evaluating sample:3 --------
    -------- evaluating sample:4 --------

Results are reported on a dataframe, that can be processed later.

``` python
results_dataframe = results_dataframe.sort_values(by=['occurences'], ascending=False)
results_dataframe.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ast_element</th>
      <th>occurences</th>
      <th>jaccard</th>
      <th>sorensen_dice</th>
      <th>levenshtein</th>
      <th>jaccard_avg</th>
      <th>sorensen_dice_avg</th>
      <th>levenshtein_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>identifier</td>
      <td>115</td>
      <td>((0.9840425531914894, 1.0, 0.9692307692307692,...</td>
      <td>((0.9919571045576407, 1.0, 0.984375, 1.0, 1.0)...</td>
      <td>((0.9840425531914894, 1.0, 0.9692307692307692,...</td>
      <td>(0.991, 0.961, 0.95)</td>
      <td>(0.995, 0.98, 0.974)</td>
      <td>(0.991, 0.955, 0.95)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>)</td>
      <td>29</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (0.994623655913978...</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (0.997304582210242...</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (0.994623655913978...</td>
      <td>(1.0, 0.992, 0.993)</td>
      <td>(1.0, 0.996, 0.996)</td>
      <td>(1.0, 0.995, 0.992)</td>
    </tr>
    <tr>
      <th>78</th>
      <td>(</td>
      <td>29</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>(1.0, 0.997, 0.987)</td>
      <td>(1.0, 0.999, 0.993)</td>
      <td>(1.0, 0.997, 0.99)</td>
    </tr>
    <tr>
      <th>121</th>
      <td>.</td>
      <td>28</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>(1.0, 0.996, 0.989)</td>
      <td>(1.0, 0.998, 0.995)</td>
      <td>(1.0, 0.996, 0.989)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>"</td>
      <td>28</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1....</td>
      <td>(1.0, 1.0, 1.0)</td>
      <td>(1.0, 1.0, 1.0)</td>
      <td>(1.0, 1.0, 1.0)</td>
    </tr>
  </tbody>
</table>
</div>

## Visualizing the Results

Once you have the dataframe with the results of the evaluation, you can
perform visualization tasks to analyse the data.

``` python
results_dataframe = results_dataframe.drop('jaccard',axis=1)
results_dataframe = results_dataframe.drop('sorensen_dice',axis=1)
results_dataframe = results_dataframe.drop('levenshtein',axis=1)
results_dataframe.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ast_element</th>
      <th>occurences</th>
      <th>jaccard_avg</th>
      <th>sorensen_dice_avg</th>
      <th>levenshtein_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>identifier</td>
      <td>115</td>
      <td>(0.991, 0.961, 0.95)</td>
      <td>(0.995, 0.98, 0.974)</td>
      <td>(0.991, 0.955, 0.95)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>)</td>
      <td>29</td>
      <td>(1.0, 0.992, 0.993)</td>
      <td>(1.0, 0.996, 0.996)</td>
      <td>(1.0, 0.995, 0.992)</td>
    </tr>
    <tr>
      <th>78</th>
      <td>(</td>
      <td>29</td>
      <td>(1.0, 0.997, 0.987)</td>
      <td>(1.0, 0.999, 0.993)</td>
      <td>(1.0, 0.997, 0.99)</td>
    </tr>
    <tr>
      <th>121</th>
      <td>.</td>
      <td>28</td>
      <td>(1.0, 0.996, 0.989)</td>
      <td>(1.0, 0.998, 0.995)</td>
      <td>(1.0, 0.996, 0.989)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>"</td>
      <td>28</td>
      <td>(1.0, 1.0, 1.0)</td>
      <td>(1.0, 1.0, 1.0)</td>
      <td>(1.0, 1.0, 1.0)</td>
    </tr>
  </tbody>
</table>
</div>

You can go to ./experimental_notebooks/result_visualizer.ipynb for a
complete example