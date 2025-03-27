from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from rules import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    forbidden_questions = load_dataset('walledai/ForbiddenQuestions', split='train')
    print(forbidden_questions['prompt'][:10])

def add_rule(example):
    rule = None
    match example['category']:
        case 'Illegal Activity':
            rule = generate_illegal_activity_rule()
        case 'Hate Speech':
            rule = generate_hate_speech_rule()
        case 'Malware':
            rule = generate_malware_rule()
        case 'Physical Harm':
            rule = generate_physical_harm_rule()
        case 'Economic Harm':
            rule = generate_economic_harm_rule()
        case 'Pornography':
            rule = generate_pornography_rule()
        case 'Political Lobbying':
            rule = generate_political_lobbying_rule()
        case 'Privacy Violence':
            rule = generate_privacy_violence_rule()
        case 'Legal Opinion':
            rule = generate_legal_opinion_rule()
        case 'Financial Advice':
            rule = generate_financial_advice_rule()
        case 'Health Consultation':
            rule = generate_health_consultation_rule()
        case 'Gov Decision':
            rule = generate_gov_decision_rule()
        case _:
            print(f'{example['category']} not handled')
            raise Exception()


if __name__ == "__main__":
    main()