import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import random

MODEL_NAME = 'chandar-lab/NeoBERT'
PAD_IDX = 0
MAX_LENGTH = 4096 

class RuleBERTDataset(Dataset):

    def __init__(self, data_folder, split):
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True) 
        print(self.tokenizer.sep_token)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        dataset_path = data_folder + f'-{split}'
        print('Dataset path: ' + dataset_path)
        data = load_from_disk(dataset_path)
        #Tokenization and padding is done all at once
        data = data.map(tokenize_and_format, batched=False, fn_kwargs={'tokenizer': tokenizer})
        # print(max(len(example['input_ids']) for example in data))
        for example in random.sample(data['input_ids'], 1):
            print(str(example))
        data.set_format(
            type="torch", 
            columns=["input_ids", 'attention_mask', "label"]
        )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        return example['input_ids'], example['attention_mask'], example['label'].float()

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
    # Tokenize the rule and prompt text, returning tensors of shape [1, sequence_length]
    rule_inputs = tokenizer(example['rule'], add_special_tokens=False, padding='do_not_pad', max_length=MAX_LENGTH, truncation=True)
    rule_tokens = torch.tensor(rule_inputs['input_ids'], dtype=torch.int).unsqueeze(0)
    prompt_inputs = tokenizer(example['prompt'], add_special_tokens=False, padding='do_not_pad', max_length=MAX_LENGTH, truncation=True)
    prompt_tokens = torch.tensor(prompt_inputs['input_ids'], dtype=torch.int).unsqueeze(0)
    
    # Create tensors for the beginning and separator tokens. 
    # Note: We wrap these in a batch dimension so they have shape [1, 1]
    bos_tensor = torch.tensor([[tokenizer.cls_token_id]])
    sep_tensor = torch.tensor([[tokenizer.sep_token_id]])
    eos_tensor = torch.tensor([[tokenizer.eos_token_id]])
    
    # cls_token = [tokenizer.cls_token_id]
    # sep_token = [tokenizer.sep_token_id]
    # tokens = cls_token + rule_tokens + sep_token + prompt_tokens
    # Concatenate all parts along the sequence dimension (dim=1)
    tokens = torch.cat([bos_tensor, rule_tokens, sep_tensor, prompt_tokens, eos_tensor])
    
    # Truncate tokens to MAX_LENTH if necessary
    tokens = tokens[:, :MAX_LENGTH]
    # For now, this will not be used
    attention_mask = (tokens != tokenizer.pad_token_id) 
    
    example['input_ids'] = tokens
    example['attention_mask'] = attention_mask.squeeze(0)  # Remove the batch dimension
    return example

def dynamic_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    # Each item in the batch is a tuple: (input_ids, attention_mask, label).
    input_ids_list, attention_mask_list, labels = zip(*batch)
    
    # Pad the sequences on the fly.
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)  # Assuming 0 is the pad token.
    padded_attention_masks = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
    # Convert labels to tensor (if they aren’t already).
    labels = torch.tensor(labels)
    return padded_input_ids, padded_attention_masks, labels

def trim_trailing_zeros_1d(tensor):
    # Find indices where tensor is nonzero
    nonzero_indices = torch.nonzero(tensor, as_tuple=True)[0]
    if nonzero_indices.numel() == 0:
        # In case the tensor is all zeros, you could choose to return an empty tensor or the original
        return tensor
    # Get the last nonzero index
    last_nonzero = nonzero_indices[-1].item()
    # Slice the tensor up to (and including) the last nonzero element
    return tensor[:last_nonzero + 1]

# def normal_collate_fn(batch):
#     '''
#     Collation function to perform dynamic padding for training and evaluation with the
#     development or validation set.

#     Inputs:
#         * batch (List[Any]): batch is a list of length batch_size, where each index contains what
#                              the dataset __getitem__ function returns.

#     Returns: To be compatible with the provided training loop, you should be returning
#         * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
#         * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
#         * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
#         * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
#         * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
#     '''
#     encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     decoder_inputs = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     decoder_targets = pad_sequence([item[3] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     # initial_decoder_inputs are scalars, so we can simply create a tensor from the list
#     initial_decoder_inputs = torch.tensor([item[4] for item in batch])
#     return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

# def test_collate_fn(batch):
#     '''
#     Collation function to perform dynamic padding for inference on the test set.

#     Inputs:
#         * batch (List[Any]): batch is a list of length batch_size, where each index contains what
#                              the dataset __getitem__ function returns.

#     Recommended returns: 
#         * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
#         * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
#         * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
#     '''
#     encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
#     initial_decoder_inputs = torch.tensor([item[2] for item in batch])
#     return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, dataset, split):
    shuffle = split == "train"

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dynamic_collate_fn)
    return dataloader

def load_forbidden_questions_data():
    datasets = [] 
    for split in ['train', 'dev', 'test']:
        data_folder = os.path.join('data/forbidden_questions')
        datasets.append(RuleBERTDataset(data_folder, split))
    
    #Horrid code
    return datasets[0], datasets[1], datasets[2]