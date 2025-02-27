import torch
import pdb
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import pandas as pd
from utils import get_model_identifiers_from_yaml, add_dataset_index

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100
    
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
        
class TextForgetDatasetQA2(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="att_"):
        super(TextForgetDatasetQA2, self).__init__()
        self.tokenizer = tokenizer 
        self.max_length = max_length
        
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        
        data_f=pd.DataFrame(self.retain_data).iloc[400:].reset_index(drop=True) # seperate 400 data point for evaluations
        self.retain_data_train = datasets.Dataset.from_pandas(data_f)
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "../tofu/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
            
        ############### from qz
        elif 'att_' in self.loss_type:
            attention_words = torch.load('../tofu_attention/attention_idx' + split + '.pth')
            if len(attention_words) != len(self.forget_data): 
                raise RuntimeError('The lengths of attention words do not match the dataset!')
            self.forget_data = self.forget_data.add_column('critical_word', [attention_words[_] for _ in attention_words])
            self.split1, self.split2 = "forget", "retain"
        ###############
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            if data_type == "retain":
                data = self.retain_data_train
                idx = (idx + torch.randint(0, len(self.retain_data_train), (1,)).item()) % len(self.retain_data_train)
            else:
                data=self.forget_data
                idx=idx
            
            question = data[idx]['question']
            answer = data[idx]['answer']
            if data_type == "idk":
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
            
            ############### from qz , here we have a copy of convert_raw_data_to_model_format, just looking to those with if 'att_' in self.loss_type:
            question_start_token, question_end_token, answer_token = self.model_configs['question_start_tag'], self.model_configs['question_end_tag'], self.model_configs['answer_tag']
            new_question = question_start_token + question + question_end_token
            new_answer = answer_token + answer
            full_text = new_question + new_answer
            num_question_tokens = len(self.tokenizer.tokenize(new_question, add_special_tokens=True))
            #print(num_question_tokens)
            if data_type=="forget":
                if 'att_' in self.loss_type:
                    attention_word=self.forget_data[idx]['critical_word']
                    asciied_answer = [''.join([_ for _ in __ if _.isascii()]) for __ in self.tokenizer.tokenize(new_answer)]
                    critical_idx_tokens = [num_question_tokens + idx for idx, _ in enumerate(asciied_answer) if _ in attention_word and _ != '' and (len(_)>=2 or _.isnumeric())]
                #print(len(self.tokenizer.tokenize(new_answer)))
                #print(len(asciied_answer))
                #print(critical_idx_tokens)
            
            encoded = self.tokenizer(
                full_text, 
                add_special_tokens=True, 
                max_length=self.max_length, 
                truncation=True, 
                )
            
            pad_length = self.max_length - len(encoded.input_ids)
            pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
            pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
            if len(encoded.input_ids) == self.max_length:
                label = encoded.input_ids
            else:
                label = encoded['input_ids'] + [self.tokenizer.eos_token_id] + [-100] * (pad_length-1)

            #change label to -100 for question tokens
            for i in range(num_question_tokens): label[i] = -100
            #print(label)
            if data_type=="forget":
                if 'att_' in self.loss_type: 
                    for idx, ele in enumerate(label): 
                        if idx not in critical_idx_tokens: label[idx] = -100  
            #print(label)
            converted_data = torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
            rets.append(converted_data)
        return rets



from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import random

class WMDPUnlearnDataset(Dataset):
    def __init__(self, tokenizer):
        super(WMDPUnlearnDataset, self).__init__()
        self.tokenizer = tokenizer

        # Load and preprocess the datasets
        self.retain_data = self.load_and_preprocess("retain")
        self.forget_data = self.load_and_preprocess("forget")

    def load_and_preprocess(self, dataset_type):
        # Define dataset paths based on type
        dataset_paths = {
            "retain": [
                ("cais/wmdp-corpora", "cyber-retain-corpus"),
                ("cais/wmdp-corpora", "bio-retain-corpus")
            ],
            "forget": [
                ("cais/wmdp-corpora", "cyber-forget-corpus"),
                ("json", "./data/wmdp/wmdp-corpora/bio_remove_dataset.jsonl")
            ]
        }

        # Load and concatenate datasets
        datasets = []
        for path, dataset_name in dataset_paths[dataset_type]:
            if path == "json":
                loaded_dataset = load_dataset("json", data_files=dataset_name, split="train", cache_dir="./.cache")
            else:
                loaded_dataset = load_dataset(path, dataset_name, cache_dir="./.cache")["train"]
            datasets.append(loaded_dataset)

        concatenated = concatenate_datasets(datasets)

        # Preprocess and set the dataset format
        processed_dataset = concatenated.map(self.preprocess, batched=True, remove_columns=concatenated.column_names)
        processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return processed_dataset

    def preprocess(self, examples):
        # Apply tokenization and potentially other transformations
        tokenized = self.tokenizer(
            examples['text'],
            max_length=1024,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        }

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        forget_sample = self.forget_data[idx]
        retain_sample = self.retain_data[random.randint(0, len(self.retain_data) - 1)]  # random sampling for retain data
        return forget_sample, retain_sample



def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def model_mix(model,before,after,update_ratio):
    for name,parameter in model.named_parameters():
        parameter.data=update_ratio*before[name[:]].cuda()+(1-update_ratio)*after[name[:]].cuda()
    return model    

