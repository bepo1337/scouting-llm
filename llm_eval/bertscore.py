from typing import List
from evaluate import load

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)

def print_bert_scores(bert_scores, results_filename):
    print("\n----- BERTScore -----")
    with open(results_filename, "a") as file:
        value_count = len(bert_scores["precision"])
        print("BERTScore number of values: ", value_count)
        file.write(f"BERTScore number of values: {value_count}\n")
        print(f"BERTScore avg. precision: {get_average(bert_scores['precision'])}\n")
        file.write(f"BERTScore avg. precision: {get_average(bert_scores['precision'])}\n")
        print(f"BERTScore avg. recall: {get_average(bert_scores['recall'])}\n")
        file.write(f"BERTScore avg. recall: {get_average(bert_scores['recall'])}\n")
        print(f"BERTScore avg. F1 score: {get_average(bert_scores['f1'])}\n")
        file.write(f"BERTScore avg. F1 score: {get_average(bert_scores['f1'])}\n")
        print(f"BERTScore hashcode: {bert_scores['hashcode']}\n")
        file.write(f"BERTScore hashcode: {bert_scores['hashcode']}\n")


def apply_bertscore(predictions, references, results_filename):
    bertscore = load("bertscore")
    # TODO microsoft/deberta-xlarge-mnli is best model to use (https://github.com/Tiiiger/bert_score#readme)
    # other model such as "roberta-large" is better, but larger obv (distilbert... takes 268MB vs roberta-large is 1.4GB)
    bert_scores = bertscore.compute(predictions=predictions, references=references,
                                    model_type="distilbert-base-uncased")
    print_bert_scores(bert_scores, results_filename)