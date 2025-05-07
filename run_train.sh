#!/bin/bash

# Array of Whisper model variants to loop through
model_variants=(
    "tiny" 
    # "base" 
    # "small" 
    "medium" 
    # "large-v2" 
    "large-v3"
    "large-v3-turbo"
    )

# Dataset name and language (common parameters)
dataset_name="irasalsabila/jvsu_asr"
language="su"
output_dir="saved_models/"
audio_noise_dir="noise_audioset_train/"

# Loop through each model variant
for variant in "${model_variants[@]}"; do
  # Construct the model name string
  model_name="openai/whisper-${variant}"

  # Construct the output directory based on the model variant
  output_model_dir="${output_dir}/whisper-${variant}-${language}"

  # Construct the full command
  command="python train.py --dataset_name \"${dataset_name}\" --language \"${language}\" --model_name \"${model_name}\" --output_dir \"${output_model_dir}\""
  # command="python train_with_noise.py --dataset_name \"${dataset_name}\" --language \"${language}\" --model_name \"${model_name}\" --output_dir \"${output_model_dir}\" --add_noise --noise_dir \"${audio_noise_dir}\""

  # Print the command (optional, but useful for debugging)
  echo "Running: $command"

  # Execute the command
  eval "$command"

  # Check the exit status of the command
  if [ $? -ne 0 ]; then
    echo "Error: Command failed for model variant: ${variant}"
    exit 1 # Exit the script with an error code if a command fails
  fi

  echo "Training complete for model variant: ${variant}"
  echo "--------------------------------------------------"
done

echo "All training runs completed."
exit 0