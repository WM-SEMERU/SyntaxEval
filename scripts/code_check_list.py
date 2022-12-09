from datasets import load_dataset 
from CodeCheckList.evaluator import Evaluator
from CodeCheckList import loader
import CodeCheckList.utils as utils


########## YOU NEED TO SET THIS FIRST #######
checkpoint = "huggingface/CodeBERTa-small-v1"
number_of_samples = 50
number_of_predictions_per_sample = 3
python_language = "python"
save_path = "output/"
########## YOU NEED TO SET THIS FIRST #######


languages = [python_language]
loader.download_grammars(languages)
evaluator = Evaluator(checkpoint, python_language)

max_token_number = evaluator.tokenizer.tokenizer.max_len_single_sentence

print('---STARTED EVALUATION----')

### PILAS A ESTO 
test_set = utils.get_random_sub_set_test_set(utils.get_test_sets(load_dataset("code_search_net", split='test'), python_language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer), number_of_samples)
#test_set = utils.get_test_sets(load_dataset("code_search_net", split='test'), python_language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer)

results_dataframe = evaluator(test_set, number_of_predictions_per_sample)
results_dataframe = results_dataframe.sort_values(by=['occurences'], ascending=False)

print(results_dataframe.head())

print('---END EVALUATION----')
print('---SAVING TABLE----')

## PILAS A ESTO 
results_dataframe.to_csv(save_path+checkpoint.replace("/","-")+"_"+str(number_of_samples)+"_"+str(number_of_predictions_per_sample)+".csv")
#results_dataframe.to_csv(save_path+checkpoint.replace("/","-")+"_"+"all"+"_"+str(number_of_predictions_per_sample)+".csv")

print('---FINISH :D----')