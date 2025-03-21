# ASR System - NLP703

A robust Automatic Speech Recognition system for processing audio with crowd noise.

## Download Links for Noise Audio Files

The audio files for crowd noise can be downloaded here:

- [Crowd Noise Audio Files](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_airlangga_mbzuai_ac_ae/Ej6C5Ygugp9IgLINY3gJXLQBwiZ-GPIBvh2LFf6TKhViAA?e=IVlFBg)
- [Vehicle City Audio Files](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/salsabila_pranida_mbzuai_ac_ae/EmoCb8_Ju6JKtuxutcRI6LgBZwEiVBemtSscwOon3xp11w?e=RoAyAz)


## How to Run the Evaluation Script

There are two ways to run the evaluation on the regular dataset:

```python
# Standard evaluation
python evaluation.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang>
```

```python
# Alternative evaluation script
python evaluation_withnoise.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang>
```

## Testing with Noise

To run the evaluation on noisy data:

### Test with random white noise
```python
python evaluation_withnoise.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang> --add_noise --noise_min <noise_min> --noise_max <noise_max>
```

### Test with other kinds of noise
Specify the directory containing the noisy audio files:
```python
python evaluation.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang> --add_noise --noise_dir <noise_dir> --noise_min <noise_min> --noise_max <noise_max>
```

## Parameters

- `language`: either `su` (Sundanese) or `jv` (Javanese)
- `hf_dataset_id`: one of `["irasalsabila/sundanese_asr_dataset_20k", "irasalsabila/sundanese_asr_dataset_20k"]`
- `checkpoint_path` example: `"saved_models/whisper-tiny-su/whisper-tiny-sundanese/checkpoint-1250"`
- `noise_min` and `noise_max`: control the range of random noise levels (between 0.0 and 1.0)
