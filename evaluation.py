import os
import torch
import argparse
import torchaudio
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
from peft import PeftModel, PeftConfig
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

parser = argparse.ArgumentParser(description="Fine-tune Whisper ASR model on Javanese/Sundanese")
parser.add_argument("--peft_model_path", type=str, help="Saved PEFT path")
parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Whisper model name")
parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name")
parser.add_argument("--language", type=str, choices=["jv", "su"], required=True, help="Language (jv or su)")
parser.add_argument("--task_type", type=str, default="transcribe", help="Task type: transcribe or translate")

args = parser.parse_args()

# args --> peft_model_path "test-openai-whisper-tiny-LORA"
# args --> model_id "openai/whisper-tiny"
peft_model_id = args.peft_model_path
peft_config = PeftConfig.from_pretrained(peft_model_id)

base_model = WhisperForConditionalGeneration.from_pretrained(args.model_id)  
model = PeftModel.from_pretrained(base_model, peft_model_id)
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)
tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=args.language, task=args.task_type)
processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task_type)

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

# Ensure the model is on GPU
model = model.to("cuda")  # ✅ Move entire model to CUDA
model_dtype = next(model.parameters()).dtype  # Get model dtype

# Set language-specific parameters
if args.language == "jv":
    audio_dir = "javanese_data"
    language = "javanese"
elif args.language == "su":
    audio_dir = "sundanese_data"
    language = "sundanese"
else:
    raise ValueError("Invalid language choice. Use 'jv' or 'su'.")

# Ensure LoRA layers are on GPU (sometimes required)
if hasattr(model, "base_model"):
    model.base_model.to("cuda")

MAX_LENGTH = 30 * 16000 

def load_audio(file_name):
    file_path = os.path.join(audio_dir, file_name + ".flac")
    try:
        speech, sr = torchaudio.load(file_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech)

        speech = speech.squeeze(0)  

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

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        speeches = []
        valid_labels = []
        valid_features = []

        # Load audio for each feature in the batch and keep on CPU initially
        for feature in features:
            speech = load_audio(feature["filename"])  # Load audio
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=tokenizer.bos_token_id
)

dataset = load_dataset(args.dataset_name)
test_dataset = dataset["test"]
eval_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

model.eval()
import evaluate
metric = evaluate.load("wer")
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.no_grad():  # No need for autocast since Whisper already handles mixed precision
        # Ensure correct dtype for input_features
        input_features = batch["input_features"].to("cuda").to(model_dtype)  # Convert to model dtype
        decoder_input_ids = batch["labels"][:, :4].to("cuda")

        generated_tokens = (
            model.generate(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=255,
            )
            .cpu()
            .numpy()
        )

        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )

    del generated_tokens, labels, batch
    gc.collect()

wer = 100 * metric.compute()
print(f"{wer=}")