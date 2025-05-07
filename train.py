import os
import torch
from torch.utils.data import DataLoader
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
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune Whisper ASR model on Javanese/Sundanese")
parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name")
parser.add_argument("--language", type=str, choices=["jw", "su"], required=True, help="Language (jv or su)")
parser.add_argument("--model_name", type=str, default="openai/whisper-small", help="Whisper model name")
parser.add_argument("--task_type", type=str, default="transcribe", help="Task type: transcribe or translate")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for fine-tuned model")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device for training")
parser.add_argument("--use_specaugment", action="store_true", help="Whether to use SpecAugment data augmentation")
parser.add_argument("--mask_time_prob", type=float, default=0.05, help="Probability of masking time steps")
parser.add_argument("--mask_time_length", type=int, default=10, help="Length of time masking")
parser.add_argument("--mask_feature_prob", type=float, default=0.0, help="Probability of masking features")
parser.add_argument("--mask_feature_length", type=int, default=10, help="Length of feature masking")
parser.add_argument("--mask_time_min_masks", type=int)
parser.add_argument("--mask_feature_min_masks", type=int)
args = parser.parse_args()

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set language-specific parameters
if args.language == "jw":
    audio_dir = "javanese_data"
    language = "javanese"
elif args.language == "su":
    audio_dir = "sundanese_data"
    language = "sundanese"
else:
    raise ValueError("Invalid language choice. Use 'jv' or 'su'.")

# Load dataset from Hugging Face Hub
dataset = load_dataset(args.dataset_name, language)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

model_id = args.model_name
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task=args.task_type)
processor = WhisperProcessor.from_pretrained(model_id, language=language, task=args.task_type)

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

        return speech.to(device)
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
                speeches.append(speech.cpu())  # Ensure on CPU
                valid_labels.append(feature["label"])
                valid_features.append(feature)

        if len(speeches) == 0:
            return {}

        # Stack speeches on CPU
        speeches = torch.stack(speeches)

        # Extract features while on CPU
        inputs = self.processor.feature_extractor(
            speeches.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        # Process labels on CPU
        labels = self.processor.tokenizer(
            valid_labels, 
            return_tensors="pt", 
            padding=True
        ).input_ids
        
        attention_mask = torch.ones_like(labels)
        labels = labels.masked_fill(attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        # Let the trainer handle device placement
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels
        }

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids if hasattr(pred, "label_ids") else None

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER score
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=tokenizer.bos_token_id
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    collate_fn=data_collator
)

eval_dataloader = DataLoader(
    test_dataset, 
    batch_size=args.batch_size,
    collate_fn=data_collator
)

metric = evaluate.load("wer")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.bfloat16  
)

model = WhisperForConditionalGeneration.from_pretrained(model_id, \
    quantization_config=quantization_config, device_map="auto")

# Update SpecAugment parameters if enabled
if args.use_specaugment:
    print("Enabling SpecAugment with the following parameters:")
    print(f"  mask_time_prob: {args.mask_time_prob}")
    print(f"  mask_time_length: {args.mask_time_length}")
    print(f"  mask_feature_prob: {args.mask_feature_prob}")
    print(f"  mask_feature_length: {args.mask_feature_length}")
    
    # Update the model configuration with SpecAugment settings
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = args.mask_time_prob
    model.config.mask_time_length = args.mask_time_length
    model.config.mask_feature_prob = args.mask_feature_prob
    model.config.mask_feature_length = args.mask_feature_length
    model.config.mask_feature_min_masks = args.mask_feature_min_masks
    model.config.mask_time_min_masks = args.mask_time_min_masks
    
    # Print the updated config for verification
    print(f"Updated model config: apply_spec_augment={model.config.apply_spec_augment}")
else:
    print("Training without SpecAugment")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

config = LoraConfig(r=64, 
    lora_alpha=128, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'fc1', 'fc2'],
    lora_dropout=0.05, bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=5,
    num_train_epochs=args.num_epochs,
    # eval_strategy="epoch",
    fp16=False,
    bf16=True,
    per_device_eval_batch_size=4,
    generation_max_length=128,
    logging_steps=1,
    eval_steps=2000,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


# If you want to use the dataloaders directly instead of datasets:
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)

model.config.use_cache = False 

trainer.train()

model = get_peft_model(model, config)
# print(model)
print("Model ID:", model_id) 
# print("model peft config:", model.peft_config)
# print("model peft config type: ", model.peft_config["default"].peft_type.value)  # Should print "LORA"

if isinstance(model.peft_config, dict) and "default" in model.peft_config:
    peft_type = model.peft_config["default"].peft_type.value 
else:
    peft_type = "lora" 

augment_suffix = "-specaugment" if args.use_specaugment else ""
peft_model_id = f"finetuned-lora/{model_id}-{peft_type}{augment_suffix}".replace("/", "-")
print("peft model id:", peft_model_id) 

# model.save_pretrained(peft_model_id)

print(f"Model will be saved at: {peft_model_id}")
