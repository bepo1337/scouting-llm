import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # Import der Stoppwörter
from collections import Counter
import numpy as np
import textstat  # Textstat-Bibliothek für Lesbarkeitsindizes
import os
import string

def analyze_text_statistics(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_data = [entry['text'] for entry in data]

    # Count number of reports
    num_reports = len(text_data)

    # Tokenize text and calculate token statistics
    token_lengths = []
    word_frequencies = Counter()
    punctuation_frequencies = Counter()
    min_tokens = float('inf')
    max_tokens = 0
    total_tokens = 0
    num_reports_without_text = 0

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    for text in text_data:
        tokens = word_tokenize(text)
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        token_count = len(filtered_tokens)
        total_tokens += token_count
        token_lengths.extend([len(token) for token in filtered_tokens])
        word_frequencies.update(filtered_tokens)
        punctuation_frequencies.update([char for char in tokens if char in string.punctuation])
        
        if token_count == 0:
            num_reports_without_text += 1
        
        if token_count < min_tokens:
            min_tokens = token_count
        if token_count > max_tokens:
            max_tokens = token_count

    average_token_length = np.mean(token_lengths)
    token_length_variance = np.var(token_lengths)
    
    # Remove punctuation marks from most common words
    punctuation_marks = set(string.punctuation)
    most_common_words = [(word, freq) for word, freq in word_frequencies.most_common(10) if word not in punctuation_marks]
    
    punctuation_counts = punctuation_frequencies.most_common()

    # Calculate readability index (Flesch-Kincaid Index)
    readability_index = textstat.flesch_reading_ease(" ".join(text_data))

    # Write statistics to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Number of reports: {}\n".format(num_reports))
        f.write("Number of reports without text: {}\n".format(num_reports_without_text))
        f.write("Average tokens per text: {:.2f}\n".format(total_tokens / num_reports))
        f.write("Minimum tokens per text: {}\n".format(min_tokens))
        f.write("Maximum tokens per text: {}\n".format(max_tokens))
        f.write("Average token length: {:.2f}\n".format(average_token_length))
        f.write("Token length variance: {:.2f}\n".format(token_length_variance))
        f.write("Readability index (Flesch-Kincaid): {:.2f}\n".format(readability_index))
        f.write("\n")
        f.write("Most common words:\n")
        for word, freq in most_common_words:
            f.write("{:<10}: {}\n".format(word, freq))
        f.write("Punctuation counts:\n")
        for char, count in punctuation_counts:
            f.write("{:<5}: {}\n".format(char, count))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "../../uhh_data.json")
    output_file = "statistics.txt"  # Replace with desired output file path
    analyze_text_statistics(json_file, output_file)