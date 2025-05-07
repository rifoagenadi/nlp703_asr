import os
import torch
import argparse
import torchaudio
import random
import numpy as np
import soundfile as sf
from datasets import load_dataset
from scipy.io import wavfile
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
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model, PeftConfig
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
parser.add_argument("--language", type=str, choices=["jw", "su"], required=True, help="Language (jv or su)")
parser.add_argument("--task_type", type=str, default="transcribe", help="Task type: transcribe or translate")
parser.add_argument("--noise_dir", type=str, help="Directory containing noise files (.wav) or (.mp3)")
parser.add_argument("--add_noise", action="store_true", help="Add noise to audio files during evaluation")
parser.add_argument("--target_snr", type=float, default=None, help="Target signal-to-noise ratio (if None, random values between 5-20dB will be used)")

args = parser.parse_args()

# Validate noise arguments
if args.add_noise and not args.noise_dir:
    parser.error("--add_noise requires --noise_dir to be specified")
if args.noise_dir and not args.add_noise:
    parser.error("--noise_dir requires --add_noise to be specified")

# If target_snr is provided but noise is not enabled, warn the user
if args.target_snr is not None and not args.add_noise:
    print("Warning: --target_snr is specified but --add_noise is not enabled. SNR value will be ignored.")

if args.language == "jw":
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
tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=args.language, task=args.task_type)
processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task_type)

# Ensure LoRA layers are on GPU
if hasattr(model, "base_model"):
    model.base_model.to("cuda")

MAX_LENGTH = 30 * 16000  # 30 seconds at 16kHz

def add_noise_to_numpy(audio_data, sample_rate, noise_directory, target_snr_db=None):
    """
    Add noise to an audio numpy array with random noise level for each segment.
    
    Args:
        audio_data: Numpy array of audio data
        sample_rate: Sample rate of the audio
        noise_directory: Directory containing noise files (MP3 or WAV)
        target_snr_db: Target SNR in dB. If None, a random value between 5 and 20 will be used.
    Returns:
        numpy.ndarray: Noisy audio data
    """
    import numpy as np
    import os
    import random
    from scipy.io import wavfile
    import librosa
    
    # If target SNR is not specified, choose a random value
    if target_snr_db is None:
        target_snr_db = random.uniform(5, 20)
    
    # List all audio files in the directory
    noise_files = [f for f in os.listdir(noise_directory) 
                   if (f.endswith('.mp3') or f.endswith('.wav')) and 
                   os.path.isfile(os.path.join(noise_directory, f))]
    
    if not noise_files:
        raise ValueError(f"No audio files found in noise directory: {noise_directory}")
    
    # Select a random noise file
    noise_file = os.path.join(noise_directory, random.choice(noise_files))
    
    # Load the noise file based on its extension
    if noise_file.endswith('.mp3'):
        # Use librosa for mp3 files
        noise_data, noise_sr = librosa.load(noise_file, sr=None)
    else:
        # Use scipy for wav files
        noise_sr, noise_data = wavfile.read(noise_file)
        # Convert to float if necessary
        if noise_data.dtype != np.float32:
            noise_data = noise_data.astype(np.float32)
            if noise_data.max() > 1.0:
                noise_data = noise_data / 32768.0  # Assumes 16-bit audio
    
    # Resample noise if needed
    if noise_sr != sample_rate:
        from scipy import signal
        duration = len(noise_data) / noise_sr
        new_length = int(duration * sample_rate)
        noise_data = signal.resample(noise_data, new_length)
    
    # If noise is longer than the audio, randomly select a segment
    if len(noise_data) > len(audio_data):
        start = random.randint(0, len(noise_data) - len(audio_data))
        noise_data = noise_data[start:start + len(audio_data)]
    
    # If noise is shorter than the audio, repeat the noise
    elif len(noise_data) < len(audio_data):
        # Calculate how many repetitions are needed
        repetitions = int(np.ceil(len(audio_data) / len(noise_data)))
        noise_data = np.tile(noise_data, repetitions)
        # Trim to match the original audio length
        noise_data = noise_data[:len(audio_data)]
    
    # If noise has multiple channels but audio is mono
    if len(noise_data.shape) > 1 and len(audio_data.shape) == 1:
        noise_data = np.mean(noise_data, axis=1)
    
    # Calculate the scaling factor based on the SNR formula
    # a = 10^(-SNR/10) * ||orig_audio||^2 / ||noise||^2
    audio_power = np.sum(audio_data ** 2)
    noise_power = np.sum(noise_data ** 2)
    
    # Avoid division by zero
    if noise_power == 0:
        return audio_data
    
    alpha = np.sqrt((audio_power / noise_power) * (10 ** (-target_snr_db / 10)))
    
    # Scale the noise by alpha and add it to the original audio
    noisy_audio = audio_data + alpha * noise_data
    
    return noisy_audio

def load_audio(file_name, add_noise=False, noise_dir=None, target_snr=None):
    file_path = os.path.join(audio_dir, file_name + ".flac")
    try:
        if add_noise and noise_dir:
            # For noise addition, use soundfile to read the audio first
            audio_data, sr = sf.read(file_path)
            
            # Add noise to the numpy array with random noise level
            noisy_audio = add_noise_to_numpy(audio_data, sr, noise_dir, target_snr_db=target_snr)
            
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
    target_snr: int

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
                target_snr=self.target_snr,
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
    target_snr=args.target_snr
)

# Load dataset
dataset = load_dataset(args.dataset_name, language)
test_dataset = dataset["test"]
eval_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

# Set model to evaluation mode
model.eval()

# Initialize WER metric
metric = evaluate.load("wer")
print(f"Starting evaluation {'with' if args.add_noise else 'without'} noise addition.")
if args.add_noise:
    print(f"Using noise from {args.noise_dir} with SNR noise is {args.target_snr}")

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
