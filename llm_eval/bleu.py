from typing import List

from evaluate import load

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)
def print_bleu_score(bleu_scores, filename):
    print("\n----- BLEU score -----")
    with open(filename, "a") as file:
        print(f"BLEU bleu: {bleu_scores['bleu']}\n")
        file.write(f"BLEU bleu: {bleu_scores['bleu']}\n")
        print(f"BLEU avg. precisions: {get_average(bleu_scores['precisions'])}\n")
        file.write(f"BLEU avg. precisions: {get_average(bleu_scores['precisions'])}\n")
        print(f"BLEU brevity_penalty: {bleu_scores['brevity_penalty']}\n")
        file.write(f"BLEU brevity_penalty: {bleu_scores['brevity_penalty']}\n")
        print(f"BLEU length_ratio: {bleu_scores['length_ratio']}\n")
        file.write(f"BLEU length_ratio: {bleu_scores['length_ratio']}\n")


def apply_bleu_score(predictions, references, filename):
    bleu = load("bleu")
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    print_bleu_score(bleu_scores, filename)