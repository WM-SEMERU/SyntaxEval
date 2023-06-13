from datasets import load_dataset 
from CodeCheckList.evaluator import Evaluator
from CodeCheckList import loader
import CodeCheckList.utils as utils
import json


########## YOU NEED TO SET THIS FIRST #######
checkpoint = "huggingface/CodeBERTa-small-v1"

masking_rate = 100/100
gpu_available = True
python_language = "python"
save_path = "/workspaces/CodeCheckList/data/linguistic_capabilities/"+checkpoint.replace("/","-")+"_"+str(masking_rate*100)+".csv"

concepts = ['for_statement', 'while_statement', 'return_statement', 'if_statement', 
            'comparison_operator', 'boolean_operator', 'for_in_clause', 'if_clause', 'identifier' ,'string', 'parameters'] #11
random_sampling = 15

################ GALERAS PATHS
galeras_paths = [
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/airflow/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/AliceMind-Baba/dataset17.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/allura/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/automated-interpretability/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/bigtop/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/cassandra/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/chatgpt-retrieval-plugin/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/cherrypy/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/consoleme/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/detect-secrets/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/discord/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/dispatch/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/django/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/django/data_2.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/django/data_3.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/django/data_4.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/django/data_5.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/dumb-init/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/EasyRec-Baba/dataset16.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/fastapi/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/FederatedScope-Baba/dataset18.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/flask/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/gpt-discord-bot/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/hubcommander/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/incubator-liminal/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/infrastructure-boxer/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/kafka/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/lemur/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/metaflow/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/nbdev/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/numpy/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/openai-python/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/opencv/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/orange3/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/paasta/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/panda3d/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/pandas/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/pandas/data_2.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/plugins-quickstart/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/pygame/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/pyqt/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/pytorch/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/qgis/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/scikit-learn/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/shap-e/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/spark/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/spyder/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/spyder/data_2.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/tensorflow/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/torchvision/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/tornado/data_1.json',
    '/workspaces/CodeCheckList/semeru-datasets/galeras_curated_raw/whisper/data_1.json'
]

################ LOAD GRAMMAR
languages = [python_language]
loader.download_grammars(languages)

################ DEFINE EVALUATOR 
evaluator = Evaluator(checkpoint, python_language, gpu_available, save_path)
max_token_number = evaluator.tokenizer.tokenizer.max_len_single_sentence

print('--- GALERAS STARTED ----')

################ TEST SET
test_set = []
for galeras_path in galeras_paths:
    test_set += json.load(open(galeras_path,))

test_set = utils.get_test_sets_galeras(test_set, python_language, max_token_number, evaluator.tokenizer)
#test_set = utils.get_random_sub_set_test_set(utils.get_test_sets(load_dataset("code_search_net", split='test'), python_language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer), number_of_samples)

print('--- GALERAS FINISHED ----')
################ CALL EVALUATOR
print('--- EVALUATION STARTED ----')

results_dataframe = evaluator(test_set, concepts, masking_rate, 'code', random_sampling)

print(results_dataframe.head())

print('--- EVALUATION FINISHED ---')