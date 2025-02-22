#!/bin/bash

# Check if language argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <language>"
  echo "Available languages: jv (Javanese), su (Sundanese)"
  exit 1
fi

LANGUAGE=$1  # Argument from user

# Set URL and directories based on language
if [ "$LANGUAGE" == "jv" ]; then
  BASE_URL="https://www.openslr.org/resources/35/asr_javanese_"
  OUTPUT_DIR="javanese_zips"
  UNPACKED_DIR="javanese_data"
elif [ "$LANGUAGE" == "su" ]; then
  BASE_URL="https://www.openslr.org/resources/36/asr_sundanese_"
  OUTPUT_DIR="sundanese_zips"
  UNPACKED_DIR="sundanese_data"
else
  echo "Error: Unsupported language '$LANGUAGE'. Choose 'jv' or 'su'."
  exit 1
fi

# 1. Download utt_spk_text.tsv (common file for both languages)
FILE_URL_UTT="https://www.openslr.org/resources/35/utt_spk_text.tsv"
OUTPUT_FILENAME_UTT="utt_spk_text.tsv"

if [ ! -f "$OUTPUT_FILENAME_UTT" ]; then
  wget -O "$OUTPUT_FILENAME_UTT" "$FILE_URL_UTT"
  if [ $? -eq 0 ]; then
    echo "File '$OUTPUT_FILENAME_UTT' downloaded successfully."
  else
    echo "Error: Failed to download file from '$FILE_URL_UTT'."
    exit 1
  fi
else
  echo "File '$OUTPUT_FILENAME_UTT' already exists, skipping download."
fi

# 2. Download and Unpack ASR files
mkdir -p "$OUTPUT_DIR"
mkdir -p "$UNPACKED_DIR"

for i in {0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f}; do
  FILE_URL="${BASE_URL}${i}.zip"
  OUTPUT_FILENAME="$OUTPUT_DIR/asr_${LANGUAGE}_${i}.zip"

  # Check if the file exists before downloading
  if [ ! -f "$OUTPUT_FILENAME" ]; then
    wget -O "$OUTPUT_FILENAME" "$FILE_URL"
    if [ $? -eq 0 ]; then
      echo "File '$OUTPUT_FILENAME' downloaded successfully."
    else
      echo "Error: Failed to download file from '$FILE_URL'."
      exit 1
    fi
  else
    echo "File '$OUTPUT_FILENAME' already exists, skipping download."
  fi

  # Unzip with overwrite option (-o)
  unzip -o -d "$UNPACKED_DIR" "$OUTPUT_FILENAME"
  if [ $? -eq 0 ]; then
    echo "File '$OUTPUT_FILENAME' unpacked successfully to '$UNPACKED_DIR'."
  else
    echo "Error: Failed to unpack file '$OUTPUT_FILENAME'."
    exit 1
  fi

  # OPTIONAL: Remove the zip file after extraction
  # rm "$OUTPUT_FILENAME"
done

echo "All files for '$LANGUAGE' downloaded and unpacked successfully."
exit 0


# bash download_data.sh jv  # For Javanese ASR dataset
# bash download_data.sh su  # For Sundanese ASR dataset
