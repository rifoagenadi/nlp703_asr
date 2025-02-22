import os
import pandas as pd
import re

def load_data(num_samples=None):
    data = pd.read_csv('utt_spk_text.tsv', sep='\t', header=None)
    return data.sample(num_samples, random_state=1312) if num_samples else data

from jiwer import wer

def calculate_wer(hypotheses, references):
    """
    Calculates Word Error Rate (WER) from hypothesis and reference text files.

    Args:
        hypotheses [str]
        references [str]

    Returns:
        float: The Word Error Rate (WER).
    """

    try:
        
        hypotheses = [re.sub(r'\s+', ' ', word.lower().strip()) for word in hypotheses]
        references = [re.sub(r'\s+', ' ', word.lower().strip()) for word in references]

        return wer(hypotheses, references)

    except FileNotFoundError:
        print("Error")
        return None  # Or raise the exception, depending on your needs
