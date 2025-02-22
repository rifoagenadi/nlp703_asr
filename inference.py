import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import argparse
from utils import load_data, calculate_wer
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Transcribe audio using a Whisper model.")
parser.add_argument("--model_name", type=str, default="openai/whisper-base", help="Name of the Whisper model to use")
parser.add_argument("--num_samples", type=int, default=20000, help="Number of samples to transcribe")
parser.add_argument("--language", type=str, default='su', help="Language code (su: Sundanese, jv: Javanese)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
args = parser.parse_args()

# Set data directory based on language
if args.language == 'su':
    data_dir = 'sundanese_data'
    language = 'sundanese'
elif args.language == 'jv':
    data_dir = 'javanese_data'
    language = 'javanese'

# Load Whisper model and processor
model_id = args.model_name
processor = AutoProcessor.from_pretrained(model_id)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load data
from sklearn.model_selection import train_test_split
if args.num_samples:
    data = load_data(data_dir=data_dir,num_samples=args.num_samples)
    _, data = train_test_split(data, test_size=0.2, random_state=1312) 
    print("Num Test Data: ", data.shape[0])
else:
    data = load_data(data_dir=data_dir)
    _, data = train_test_split(data, test_size=0.2, random_state=1312) 
file_names = data[0].tolist()
labels = data[2].tolist()

def process_batch(file_paths, processor, target_sample_rate=16000):
    """
    Process a batch of audio files and prepare them for the model.
    
    Args:
        file_paths: List of audio file paths
        processor: Whisper processor
        target_sample_rate: Target sampling rate for audio
    
    Returns:
        Dictionary containing batched input features
    """
    max_length = 0
    batch_inputs = []
    
    # First pass: load all audio and find maximum length
    for file_path in file_paths:
        try:
            waveform, sample_rate = torchaudio.load(f'{data_dir}/{file_path}.flac')
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
            max_length = max(max_length, waveform.shape[0])
            batch_inputs.append(waveform)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Second pass: pad all inputs to max_length
    padded_inputs = []
    for waveform in batch_inputs:
        if waveform.shape[0] < max_length:
            padding = torch.zeros(max_length - waveform.shape[0])
            waveform = torch.cat([waveform, padding])
        padded_inputs.append(waveform)

    # Stack all inputs and process
    if padded_inputs:
        input_features = processor(
            audio=torch.stack(padded_inputs).numpy(),
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return input_features
    return None

# Process in batches
from tqdm.auto import tqdm
predictions = []
num_files = len(file_names)
batch_size = args.batch_size

for i in tqdm(range(0, num_files, batch_size)):
    batch_files = file_names[i:i + batch_size]
    inputs = process_batch(batch_files, processor)
    
    if inputs is not None:
        with torch.no_grad():
            input_features = inputs.input_features.to(device)
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend(transcriptions)

# Calculate and print results
# print("Predictions:", predictions)
# print("Labels:", labels)
wer = calculate_wer(predictions, labels)
print("WER:", wer*100)