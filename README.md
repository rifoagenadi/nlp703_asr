# ASR System - NLP703

A robust Automatic Speech Recognition system for processing audio with crowd noise.

## Download Links for Noise Audio Files

The audio files for crowd noise can be downloaded here:

- [Vehicle City Audio Files](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_airlangga_mbzuai_ac_ae/Ej6C5Ygugp9IgLINY3gJXLQBwiZ-GPIBvh2LFf6TKhViAA?e=IVlFBg)
- [Crowd Noise Audio Files]()


## How to Run the Evaluation Script
There are two ways to run the evaluation on the regular dataset
`python evaluation.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang>` or `python evaluation_withnoise.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang>`


To run the evaluation on noisy set:
- Test on data with random white noise `python evaluation_withnoise.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang> --add_noise --noise_min <noise_min> --noise_max <noise_max>`
- Test on data with other kind of noise, specify the directory containing the noisy audio files `python evaluation.py --peft_model_path <checkpoint_path> --model_id <model_id> --dataset_name <hf_dataset_id> --language <lang> --add_noise --noise_dir <noise_dir> --noise_min <noise_min> --noise_max <noise_max>`

- language is either `su` or `jv`
- the regular dataset `hf_dataset_id` is one of `["irasalsabila/sundanese_asr_dataset_20k", "irasalsabila/sundanese_asr_dataset_20k"]`
- `checkpoint_path` example: `"saved_models/whisper-tiny-su/whisper-tiny-sundanese/checkpoint-1250"`
