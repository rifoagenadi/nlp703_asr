from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

# args --> peft_model_path "test-openai-whisper-tiny-LORA"
# args --> model_id "openai/whisper-tiny"
args.model_name = args.peft_model_path
peft_model_id = peft_model_path
peft_config = PeftConfig.from_pretrained(peft_model_id)

base_model = WhisperForConditionalGeneration.from_pretrained(args.model_id)  
model = PeftModel.from_pretrained(base_model, peft_model_id)

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

# Ensure the model is on GPU
model = model.to("cuda")  # ✅ Move entire model to CUDA
model_dtype = next(model.parameters()).dtype  # Get model dtype

# Ensure LoRA layers are on GPU (sometimes required)
if hasattr(model, "base_model"):
    model.base_model.to("cuda")

eval_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

model.eval()
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