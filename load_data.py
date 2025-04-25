import os
from statistics import mean
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
import torch.nn as nn
import random #I'm not using it now, but I often find it useful for debugging

MODEL_NAME = 'chandar-lab/NeoBERT'
PAD_IDX = 0
MAX_LENGTH = 4096 

class RuleBERTDataset(Dataset):

    def __init__(self, data_folder, split):
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True) 
        # print(self.tokenizer.sep_token)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        dataset_path = data_folder + f'-{split}'
        print('Dataset path: ' + dataset_path)
        data = load_from_disk(dataset_path)
        print(mean(data['label']))
        #Tokenization and padding is done all at once
        data = data.map(tokenize_and_format, batched=False, fn_kwargs={'tokenizer': tokenizer})
        # print(max(len(example['input_ids']) for example in data))
        # for example in random.sample(data['input_ids'], 1):
        #     print(str(example))
        data.set_format(
            type="torch", 
            columns=["input_ids", 'attention_mask', "label"]
        )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        return example['input_ids'], example['attention_mask'], example['label'].long()

# def tokenize_and_format(example, tokenizer):
#     rule_tokens = tokenizer.encode(example['rule'], add_special_tokens=False, return_tensors='pt')
#     prompt_tokens = tokenizer.encode(example['prompt'], add_special_tokens=False, return_tensors='pt') 
#     tokens = [tokenizer.bos_token_id] + rule_tokens + [tokenizer.sep_token_id] + prompt_tokens
#     tokens = tokens[:MAX_LENTH]
#     padding_length = MAX_LENTH - len(tokens)
#     tokens += [tokenizer.pad_token_id] * padding_length
#     attention_mask = [1] * (MAX_LENTH - padding_length) + [0] * padding_length
#     example['input_ids'] = tokens
#     example['attention_mask'] = attention_mask
#     return example

def tokenize_and_format(example, tokenizer):
    encoded = tokenizer(
        example['rule'],
        example['prompt'],
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='do_not_pad',
        return_attention_mask=True
    )
    example['input_ids']      = torch.tensor(encoded['input_ids'], dtype=torch.long)
    example['attention_mask'] = torch.tensor(encoded['attention_mask'], dtype=torch.long)
    return example


# def tokenize_and_format(example, tokenizer):
#     # Tokenize the rule and prompt text, returning tensors of shape [1, sequence_length]
#     rule_inputs = tokenizer(example['rule'], add_special_tokens=False, padding='do_not_pad', max_length=MAX_LENGTH, truncation=True)
#     rule_tokens = torch.tensor(rule_inputs['input_ids'], dtype=torch.int).unsqueeze(0)
#     prompt_inputs = tokenizer(example['prompt'], add_special_tokens=False, padding='do_not_pad', max_length=MAX_LENGTH, truncation=True)
#     prompt_tokens = torch.tensor(prompt_inputs['input_ids'], dtype=torch.int).unsqueeze(0)
    
#     # Create tensors for the beginning and separator tokens. 
#     # Note: We wrap these in a batch dimension so they have shape [1, 1]
#     bos_tensor = torch.tensor([[tokenizer.cls_token_id]])
#     sep_tensor = torch.tensor([[tokenizer.sep_token_id]])
#     eos_tensor = torch.tensor([[tokenizer.eos_token_id]])
    
#     # cls_token = [tokenizer.cls_token_id]
#     # sep_token = [tokenizer.sep_token_id]
#     # tokens = cls_token + rule_tokens + sep_token + prompt_tokens
#     # Concatenate all parts along the sequence dimension (dim=1)
#     tokens = torch.cat([bos_tensor, rule_tokens, sep_tensor, prompt_tokens, eos_tensor])
    
#     # Truncate tokens to MAX_LENTH if necessary
#     tokens = tokens[:, :MAX_LENGTH]
#     # For now, this will not be used
#     attention_mask = (tokens != tokenizer.pad_token_id) 
    
#     example['input_ids'] = tokens
#     example['attention_mask'] = attention_mask.squeeze(0)  # Remove the batch dimension
#     return example

def packing_collate_fn(batch):
    # Each sample is a tuple: (input_ids, attention_mask, label)
    input_ids_list, attention_mask_list, labels = zip(*batch)

    # Remove extra batch dim (from unsqueeze during tokenization)
    input_ids_list = [x.squeeze(0) if x.dim() == 2 else x for x in input_ids_list]
    attention_mask_list = [m.squeeze(0) if m.dim() == 2 else m for m in attention_mask_list]

    # Compute lengths per example.
    lengths = [x.size(0) for x in input_ids_list]
    total_length = sum(lengths)

    # Concatenate tokens and masks.
    packed_input_ids = torch.cat(input_ids_list, dim=0)  # shape: (total_length,)
    packed_attention_mask = torch.cat(attention_mask_list, dim=0)

    # **New**: Compute position_ids for each example individually.
    pos_ids_list = [torch.arange(l, dtype=torch.long) for l in lengths]
    packed_position_ids = torch.cat(pos_ids_list, dim=0)  # shape: (total_length,)

    # Unsqueeze to add a batch dimension.
    packed_input_ids = packed_input_ids.unsqueeze(0)
    packed_attention_mask = packed_attention_mask.unsqueeze(0)
    packed_position_ids = packed_position_ids.unsqueeze(0)

    # Build the cumulative lengths vector.
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)

    # The maximum individual sequence length.
    max_seqlen = max(lengths)
    batch_labels = torch.tensor(labels)

    return {
        'input_ids': packed_input_ids,
        'attention_mask': packed_attention_mask,
        'position_ids': packed_position_ids,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
        'labels': batch_labels
    }

# def dynamic_collate_fn(batch):
#     from torch.nn.utils.rnn import pad_sequence
#     # Each item in the batch is a tuple: (input_ids, attention_mask, label).
#     input_ids_list, attention_mask_list, labels = zip(*batch)
    
#     # Pad the sequences on the fly.
#     padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)  # Assuming 0 is the pad token.
#     padded_attention_masks = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
#     # Convert labels to tensor (if they aren’t already).
#     labels = torch.tensor(labels)
#     return padded_input_ids, padded_attention_masks, labels

# def trim_trailing_zeros_1d(tensor):
#     # Find indices where tensor is nonzero
#     nonzero_indices = torch.nonzero(tensor, as_tuple=True)[0]
#     if nonzero_indices.numel() == 0:
#         # In case the tensor is all zeros, you could choose to return an empty tensor or the original
#         return tensor
#     # Get the last nonzero index
#     last_nonzero = nonzero_indices[-1].item()
#     # Slice the tensor up to (and including) the last nonzero element
#     return tensor[:last_nonzero + 1]

def get_dataloader(batch_size, dataset, split):
    shuffle = split == "train"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=packing_collate_fn, num_workers=4, pin_memory=True)
    return dataloader

def load_forbidden_questions_data():
    datasets = [] 
    for split in ['train', 'dev', 'test']:
        data_folder = os.path.join('data/sorrybench-transformed')
        datasets.append(RuleBERTDataset(data_folder, split))
    
    for dataset in datasets:
        print(len(dataset))
    
    #Horrid code
    return datasets[0], datasets[1], datasets[2]

def load_model():
    # model = Amelia(num_ffnn_layers=config['num_layers'], dropout=config['dropout'])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, problem_type='single_label_classification', trust_remote_code=True)
    
    # for param in model.model.parameters():
    #     param.requires_grad = False
    
    # #Unfreeze the last 25% of layers
    # for layer in model.model.transformer_encoder[5:]:
    #     for p in layer.parameters():
    #         p.requires_grad = True
            
    #Leaving the top 25% of encoders unfrozen
    # for layer in model.model.transformer_encoder:
    #     for param in layer.parameters():
    #         param.requires_grad = False 
    
    # nn.init.xavier_normal_(model.classifier.weight)
    # nn.init.zeros_(model.classifier.bias)
    # nn.init.xavier_normal_(model.dense.weight)
    # nn.init.zeros_(model.dense.bias)
    return model