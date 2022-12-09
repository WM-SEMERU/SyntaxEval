# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/predictor.ipynb.

# %% auto 0
__all__ = ['Predictor']

# %% ../nbs/predictor.ipynb 2
import torch
import CodeCheckList
from .tokenizer import CodeTokenizer
from transformers import AutoModelForMaskedLM, BatchEncoding

# %% ../nbs/predictor.ipynb 3
class Predictor:
    """Predictor Module"""
    def __init__(self, tokenizer, model, gpu_available):
        self.model = model
        self.tokenizer = tokenizer
        self.gpu_available = gpu_available

    def __call__(self, masked_code_encoding: BatchEncoding, code_encoding: BatchEncoding, top_k: int):
        masked_indexes = list(map(lambda entry: entry[0],
            list(filter(lambda entry: True if entry[1] == self.tokenizer.tokenizer.mask_token_id else False, enumerate(masked_code_encoding['input_ids'])))))
        code_encoding['input_ids'][0] = torch.tensor([torch.tensor(input_id) for input_id in masked_code_encoding['input_ids']])
        tokens_tensor = code_encoding['input_ids']
        attention_mask = code_encoding['attention_mask']
        if(self.gpu_available):
            tokens_tensor = tokens_tensor.to('cuda:0')
            attention_mask = attention_mask.to('cuda:0')
        model_input = {'input_ids' : tokens_tensor,
          'attention_mask' : attention_mask}
        model_prediction = self.model(**model_input)
        preditions = []
        for k_index in range(0, top_k):
            preditions.append(code_encoding['input_ids'][0].tolist().copy())
        for masked_index in masked_indexes:
            values, predictions = model_prediction['logits'][0][masked_index].topk(top_k)
            for k_index in range(0, top_k):
                preditions[k_index][masked_index] = predictions[k_index]
        prediction_list = list(map(lambda prediction: self.tokenizer.tokenizer.decode(list(filter(lambda token_id: False if token_id == self.tokenizer.tokenizer.bos_token_id or 
                    token_id == self.tokenizer.tokenizer.eos_token_id else True, prediction))), preditions))
        return prediction_list

    @staticmethod
    def from_pretrained(
        name_or_path: str,          #name or path of the model
        tokenizer: CodeTokenizer,   #tokenizer, has to be of the same type that the pretrained model
        gpu_available = False       #indicates if gpu should be used for predictions
    ): 
        """Create a AutoModelForMaskedLM from a pretrained model."""
        model = AutoModelForMaskedLM.from_pretrained(name_or_path)
        if(gpu_available):
            print("------------------Loading Model into GPU------------------")
            model = AutoModelForMaskedLM.from_pretrained(name_or_path).to('cuda:0')
        return Predictor(tokenizer, model, gpu_available)

