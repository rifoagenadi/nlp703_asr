import os
import torch
import argparse
import torchaudio
import random
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
import evaluate
from peft import PeftModel, PeftConfig
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

parser = argparse.ArgumentParser(description="Fine-tune Whisper ASR model on Javanese/Sundanese with noise addition")
parser.add_argument("--peft_model_path", type=str, help="Saved PEFT path", default=None)
parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Whisper model name")
parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name")
parser.add_argument("--language", type=str, choices=["jv", "su"], required=True, help="Language (jv or su)")
parser.add_argument("--task_type", type=str, default="transcribe", help="Task type: transcribe or translate")
parser.add_argument("--noise_dir", type=str, help="Directory containing noise files (.wav)")
parser.add_argument("--add_noise", action="store_true", help="Add noise to audio files during evaluation")
parser.add_argument("--noise_level", type=float, default=1.0, help="Scaling factor for noise (0.0-1.0)")

args = parser.parse_args()

if args.language == "jv":
    audio_dir = "javanese_data"
    language = "javanese"
elif args.language == "su":
    audio_dir = "sundanese_data"
    language = "sundanese"
else:
    raise ValueError("Invalid language choice. Use 'jv' or 'su'.")

#peft_model_id = args.peft_model_path
#peft_config = PeftConfig.from_pretrained(peft_model_id)

base_model = WhisperForConditionalGeneration.from_pretrained(args.model_id)

# Conditionally apply PEFT if path is provided
if args.peft_model_path:
    peft_config = PeftConfig.from_pretrained(args.peft_model_path)
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
else:
    model = base_model  # Use original weights

model = model.to("cuda")
model_dtype = next(model.parameters()).dtype

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)
tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=language, task=args.task_type)
processor = WhisperProcessor.from_pretrained(args.model_id, language=language, task=args.task_type)

# Ensure LoRA layers are on GPU
if hasattr(model, "base_model"):
    model.base_model.to("cuda")

MAX_LENGTH = 30 * 16000  # 30 seconds at 16kHz

# Noise addition functions
def convert_flac_to_numpy(flac_path):
    """Convert FLAC file to numpy array with sample rate."""
    data, samplerate = sf.read(flac_path)
    return data, samplerate

def convert_wav_to_numpy(wav_path):
    """Convert WAV file to numpy array with sample rate."""
    data, samplerate = sf.read(wav_path)
    return data, samplerate

def add_noise_to_numpy(audio_data, sample_rate, noise_directory, noise_level=1.0):
    """
    Add noise to an audio numpy array.
    
    Args:
        audio_data: Numpy array of audio data
        sample_rate: Sample rate of the audio
        noise_directory: Directory containing noise files (WAV)
        noise_level: Scaling factor for noise (0.0-1.0)
        
    Returns:
        numpy.ndarray: Noisy audio data
    """
    # Get the length of the audio
    audio_length = len(audio_data)
    
    # Create an empty array for the noise with the same length as the original audio
    noise_data = np.zeros_like(audio_data)
    
    # List all noise files
    noise_files = [os.path.join(noise_directory, f) for f in os.listdir(noise_directory) 
                   if f.endswith('.wav')]
    
    # Make sure we have at least 10 noise files
    if len(noise_files) < 10:
        raise ValueError(f"Need at least 10 noise files, but only found {len(noise_files)}")
    
    # Divide the audio into 10 parts
    segment_length = audio_length // 10
    
    # For each part, randomly sample from the noise files
    for i in range(10):
        # Define the start and end indices for this segment
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < 9 else audio_length
        
        # Randomly select a noise file
        noise_file = random.choice(noise_files)
        
        # Read the noise file
        noise_segment, noise_sample_rate = convert_wav_to_numpy(noise_file)
        
        # If the noise sample rate doesn't match the audio sample rate, we would need to resample
        if noise_sample_rate != sample_rate:
            raise ValueError(f"Sample rate mismatch: audio={sample_rate}Hz, noise={noise_sample_rate}Hz")
        
        # If noise is shorter than the segment, repeat it
        if len(noise_segment) < (end_idx - start_idx):
            repeats = (end_idx - start_idx) // len(noise_segment) + 1
            noise_segment = np.tile(noise_segment, repeats)
        
        # Trim the noise to match the segment length
        noise_segment = noise_segment[:(end_idx - start_idx)]
        
        # If noise has multiple channels but audio is mono, convert noise to mono
        if len(noise_segment.shape) > 1 and len(audio_data.shape) == 1:
            noise_segment = np.mean(noise_segment, axis=1)
        
        # If audio has multiple channels but noise is mono, duplicate noise across channels
        if len(audio_data.shape) > 1 and len(noise_segment.shape) == 1:
            noise_segment = np.column_stack([noise_segment] * audio_data.shape[1])
        
        # Add the noise segment to the corresponding part of the noise data (scaled by noise_level)
        noise_data[start_idx:end_idx] = noise_segment * noise_level
    
    # Add the noise to the original audio
    noisy_audio = audio_data + noise_data
    
    return noisy_audio

def load_audio(file_name, add_noise=False, noise_dir=None, noise_level=1.0):
    file_path = os.path.join(audio_dir, file_name + ".flac")
    try:
        if add_noise and noise_dir:
            # For noise addition, use soundfile to read the audio first
            audio_data, sr = sf.read(file_path)
            
            # Add noise to the numpy array
            noisy_audio = add_noise_to_numpy(audio_data, sr, noise_dir, noise_level)
            
            # Convert numpy array back to torch tensor
            speech = torch.tensor(noisy_audio)
            
            # Check if we need to convert sample rate to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                speech = resampler(speech.unsqueeze(0)).squeeze(0)
        else:
            # Standard audio loading without noise
            speech, sr = torchaudio.load(file_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                speech = resampler(speech)
            speech = speech.squeeze(0)

        # Handle length (truncate or pad as needed)
        if speech.shape[0] > MAX_LENGTH:
            speech = speech[:MAX_LENGTH]  # Truncate
        else:
            pad = MAX_LENGTH - speech.shape[0]
            speech = torch.cat([speech, torch.zeros(pad)])  # Pad with zeros

        return speech
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None 

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    add_noise: bool
    noise_dir: str
    noise_level: float

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        speeches = []
        valid_labels = []
        valid_features = []

        # Load audio for each feature in the batch
        for feature in features:
            speech = load_audio(
                feature["filename"], 
                add_noise=self.add_noise, 
                noise_dir=self.noise_dir,
                noise_level=self.noise_level
            )
            if speech is not None:
                speeches.append(speech)
                valid_labels.append(feature["label"])
                valid_features.append(feature)

        if len(speeches) == 0:
            return {}

        # Stack speeches
        speeches = torch.stack(speeches)

        # Extract features
        inputs = self.processor.feature_extractor(
            speeches.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        # Process labels
        labels = self.processor.tokenizer(
            valid_labels, 
            return_tensors="pt", 
            padding=True
        ).input_ids
        
        attention_mask = torch.ones_like(labels)
        labels = labels.masked_fill(attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        # Return a dictionary containing 'input_features' and 'labels'
        return {
            "input_features": inputs.input_features,
            "labels": labels
        }

# Initialize data collator with noise options
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=tokenizer.bos_token_id,
    add_noise=args.add_noise,
    noise_dir=args.noise_dir,
    noise_level=args.noise_level
)

# Load dataset
dataset = load_dataset(args.dataset_name)
test_dataset = dataset["test"]
eval_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

# Set model to evaluation mode
model.eval()

# Initialize WER metric
metric = evaluate.load("wer")
print(f"Starting evaluation {'with' if args.add_noise else 'without'} noise addition.")
if args.add_noise:
    print(f"Using noise from {args.noise_dir} with level {args.noise_level}")

# Run evaluation
for step, batch in enumerate(tqdm(eval_dataloader)):
    if not batch:  # Skip empty batches (happens if all audio loading failed)
        continue
        
    with torch.no_grad():
        # Move input features to GPU with correct dtype
        input_features = batch["input_features"].to("cuda").to(model_dtype)
        decoder_input_ids = batch["labels"][:, :4].to("cuda")

        # Generate tokens
        generated_tokens = (
            model.generate(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=255,
            )
            .cpu()
            .numpy()
        )

        # Decode predictions and references
        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Add batch to WER calculation
        metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )

    # Free memory
    del generated_tokens, labels, batch
    gc.collect()

# Calculate and report WER
wer = 100 * metric.compute()
print(f"WER: {wer:.2f}%")
print(f"Evaluation completed {'with' if args.add_noise else 'without'} noise addition.")