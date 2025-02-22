import os
import pandas as pd
import re

# def load_data(num_samples=None):
#     data = pd.read_csv('utt_spk_text.tsv', sep='\t', header=None)
#     return data.sample(num_samples, random_state=1312) if num_samples else data

def load_data(data_dir=None, num_samples=None):
    """
    Load transcription data and filter based on available audio files in the given directory.
    """

    df = pd.read_csv("utt_spk_text.tsv", sep='\t', header=None)

    if not os.path.exists(data_dir):
        print(f"Warning: Audio directory '{data_dir}' not found!")
        return pd.DataFrame()  # Return an empty DataFrame

    existing_files = {f.split(".flac")[0] for f in os.listdir(data_dir) if f.endswith(".flac")}

    # Filter into available and missing
    df_available = df[df[0].isin(existing_files)].copy()

    # If sampling is requested, return a random subset
    if num_samples:
        df_available = df_available.sample(num_samples, random_state=1312)

    print("Available data loaded:", len(df_available))

    return df_available

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
