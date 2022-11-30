from datasets import load_dataset 
from CodeCheckList.evaluator import Evaluator
from CodeCheckList import loader
import CodeCheckList.utils as utils


########## YOU NEED TO SET THIS FIRST #######
checkpoint = "huggingface/CodeBERTa-small-v1"
number_of_samples = 100
number_of_predictions_per_sample = 3
python_language = "python"
save_path = "output/"
########## YOU NEED TO SET THIS FIRST #######


languages = [python_language]
loader.download_grammars(languages)
evaluator = Evaluator(checkpoint, python_language)

max_token_number = evaluator.tokenizer.tokenizer.max_len_single_sentence

test_set = load_dataset("code_search_net", split='test')
test_set = test_set.filter(lambda sample: True if sample['language']== python_language
            and len(sample['func_code_tokens']) <= max_token_number
            and len(evaluator.tokenizer.tokenizer(sample['whole_func_string'])['input_ids']) <= max_token_number else False, num_proc=1)
test_set = utils.get_sub_set_test_set(test_set, number_of_samples)

results_dataframe = evaluator(test_set, number_of_predictions_per_sample)
results_dataframe = results_dataframe.sort_values(by=['occurences'], ascending=False)

results_dataframe.to_csv(save_path+checkpoint.replace("/","-")+"_"+str(number_of_samples)+"_"+str(number_of_predictions_per_sample)+".csv")


