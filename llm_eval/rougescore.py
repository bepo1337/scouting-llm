from typing import List
from evaluate import load

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)

def print_rouge_score(rouge_scores):
    print("\n----- ROUGE score -----")
    print(f"ROGUE avg. rouge1: {get_average(rouge_scores['rouge1'])}")
    print(f"ROGUE avg. rouge2: {get_average(rouge_scores['rouge2'])}\n")
    print(f"ROGUE avg. rougeL: {get_average(rouge_scores['rougeL'])}\n")
    print(f"ROGUE avg. rougeLsum: {rouge_scores['rougeLsum']}\n")


def apply_rouge_score(predictions, references):
    rouge = load("rouge")
    # other model such as "roberta-large" is better, but larger obv (distilbert... takes 268MB vs roberta-large is 1.4GB)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    print_rouge_score(rouge_scores)