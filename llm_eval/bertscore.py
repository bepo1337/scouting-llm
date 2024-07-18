from typing import List
from evaluate import load

def get_average(float_list: List[float]) -> float:
    return sum(float_list) / len(float_list)

def print_bert_scores(bert_scores):
    print("\n----- BERTScore -----")
    #print(bert_scores)
    value_count = len(bert_scores["precision"])
    print("BERTScore number of values: ", value_count)
    print(f"BERTScore avg. precision: {get_average(bert_scores['precision'])}")
    print(f"BERTScore avg. recall: {get_average(bert_scores['recall'])}\n")
    print(f"BERTScore avg. F1 score: {get_average(bert_scores['f1'])}\n")
    print(f"BERTScore hashcode: {bert_scores['hashcode']}\n")


def apply_bertscore(predictions, references):
    bertscore = load("bertscore")
    # TODO microsoft/deberta-xlarge-mnli is best model to use (https://github.com/Tiiiger/bert_score#readme)
    # other model such as "roberta-large" is better, but larger obv (distilbert... takes 268MB vs roberta-large is 1.4GB)
    bert_scores = bertscore.compute(predictions=predictions, references=references,
                                    model_type="distilbert-base-uncased")
    print_bert_scores(bert_scores)