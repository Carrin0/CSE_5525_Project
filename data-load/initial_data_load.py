from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from rules import rule_generators
import random
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    forbidden_prompts = load_dataset('walledai/ForbiddenQuestions', split='train')
    # We want to make sure that the MAX number of rules is the same for correct and incorrect examples, knowing that
    correct_examples = get_correct_examples(forbidden_prompts, range(len(rule_generators) - 2))
    wrong_examples = get_failing_examples(forbidden_prompts, range(1, len(rule_generators) - 1))
    safe_prompts_path = os.path.join('data/safe-prompts.txt')
    safe_prompts = load_dataset('text', data_files=safe_prompts_path)
    safe_prompts = process_safe_prompts(safe_prompts, range(1, len(rule_generators) - 1))

    print('Correct Examples')
    for example in random.sample(list(correct_examples), 2):
        print(str(example) + '\n')
    
    print('\nIncorrect Examples')
    for example in random.sample(list(wrong_examples), 2):
        print(str(example) + '\n')

    transformed_data = concatenate_datasets([correct_examples, wrong_examples, safe_prompts])
    print('\nBasic stats')
    print('Num_rows' + str(len(transformed_data)))

    data_path = os.path.join('data/forbidden_questions')
    transformed_data.save_to_disk(data_path)
    transformed_data.to_csv(data_path + '.csv', index=False)

def process_safe_prompts(data, rule_nums):
    data = data['train']
    data = data.rename_column('text', 'prompt')
    datasets = []
    for num_rules in rule_nums:
        safe_dataset = data.map(lambda example: add_random_rules(example, num_rules))
        datasets.append(safe_dataset)
    safe_dataset = concatenate_datasets(datasets)
    category = ['safe'] * len(safe_dataset)
    label = [0] * len(safe_dataset)
    safe_dataset.add_column('category', category)
    safe_dataset.add_column('label', label)
    return safe_dataset



def get_correct_examples(data, diff_rule_nums):
    """Gets all examples with a positive (1) data label"""
    assert max(diff_rule_nums) <= len(rule_generators) - 1
    datasets = []

    for num_wrong_rules in diff_rule_nums:
        correct_example_set = data.map(lambda example: add_correct_rule(example, num_wrong_rules))
        datasets.append(correct_example_set)

    return concatenate_datasets(datasets)

def get_failing_examples(data, wrong_rule_nums):
    """Gets all examples with a negative (0) data label"""
    assert max(wrong_rule_nums) <= len(rule_generators) - 1
    datasets = []

    for num_wrong_rules in wrong_rule_nums:
        failing_example_set = data.map(lambda example: add_wrong_rules(example, num_wrong_rules))
        datasets.append(failing_example_set)
    
    return concatenate_datasets(datasets)
        
def add_random_rules(example, num_rules):
    categories = [key for key in rule_generators]
    rules = [rule_generators[category]() for category in random.sample(categories, num_rules)]
    random.shuffle(rules)
    example['rule'] = " ".join(rules)
    #Seems like useful data to have
    example['num rules'] = num_rules

def add_correct_rule(example, num_wrong_rules=0):
    category = example['category']
    wrong_categories = [key for key in rule_generators if key != category]

    correct_rule = [rule_generators[category]()]
    wrong_rules = [rule_generators[category]() for category in random.sample(wrong_categories, num_wrong_rules)]
    all_rules = wrong_rules + correct_rule
    #Do this to avoid any patterns in where I put the rules
    random.shuffle(all_rules)
    #Combine the set of rules into one
    example['rule'] = " ".join(all_rules)
    #Seems like useful data to have
    example['num rules'] = len(all_rules)
    #Since it does contain the correct rule, it will pass
    example['label'] = 1
    return example

def add_wrong_rules(example, num_rules=1):
    category = example['category']
    wrong_categories = [key for key in rule_generators if key != category]

    wrong_rules = [rule_generators[category]() for category in random.sample(wrong_categories, num_rules)]
    #Do this to avoid any patterns in where I put the rules
    random.shuffle(wrong_rules)
    #Combine the set of rules into one
    example['rule'] = " ".join(wrong_rules)
    #Seems like useful data to have
    example['num rules'] = len(wrong_rules)
    #Since it does contain the correct rule, it will pass
    example['label'] = 0
    return example
    

if __name__ == "__main__":
    main()