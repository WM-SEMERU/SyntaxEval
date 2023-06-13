from datasets import load_dataset 
from CodeCheckList.evaluator import Evaluator
from CodeCheckList import loader
import CodeCheckList.utils as utils
import json


########## YOU NEED TO SET THIS FIRST #######
checkpoint = "huggingface/CodeBERTa-small-v1"

masking_rate = 25/100
gpu_available = True
python_language = "python"
save_path = "/workspaces/CodeCheckList/data/linguistic_capabilities/"

test_set_path = '/workspaces/CodeCheckList/data/galeras/galeras.json'
concepts = ['for_statement', 'while_statement', 'return_statement', ']', ')', 'if_statement', 'comparison_operator', 'boolean_operator', 'for_in_clause', 'if_clause', 'list_comprehension', 'lambda', 'identifier' ,'string']

################ LOAD GRAMMAR
languages = [python_language]
loader.download_grammars(languages)

################ DEFINE EVALUATOR 
evaluator = Evaluator(checkpoint, python_language, gpu_available)
max_token_number = evaluator.tokenizer.tokenizer.max_len_single_sentence

print('--- EVALUATION STARTED ----')

################ TEST SET
test_set = utils.get_test_sets_galeras(json.load(open(test_set_path,)), python_language, max_token_number, evaluator.tokenizer)
#test_set = utils.get_random_sub_set_test_set(utils.get_test_sets(load_dataset("code_search_net", split='test'), python_language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer), number_of_samples)

################ CALL EVALUATOR
results_dataframe = evaluator(test_set, concepts, masking_rate, 'code')

print(results_dataframe.head())

print('--- EVALUATION FINISHED ---')

################ OUTPUT
#results_dataframe.to_csv(save_path+checkpoint.replace("/","-")+"_"+str(number_of_samples)+"_"+str(masking_rate*100)+"_"+str(number_of_predictions_per_sample)+".csv")
results_dataframe.to_csv(save_path+checkpoint.replace("/","-")+"_"+str(masking_rate*100)+"_"+"all"+".csv")