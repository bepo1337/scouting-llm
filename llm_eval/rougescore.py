from typing import List
from evaluate import load

def print_rouge_score(rouge_scores):
    print("\n----- ROUGE score -----")
    print(f"ROGUE rouge1: {rouge_scores['rouge1']}")
    print(f"ROGUE rouge2: {rouge_scores['rouge2']}\n")
    print(f"ROGUE rougeL: {rouge_scores['rougeL']}\n")
    print(f"ROGUE rougeLsum: {rouge_scores['rougeLsum']}\n")


def apply_rouge_score(predictions, references):
    rouge = load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    print_rouge_score(rouge_scores)