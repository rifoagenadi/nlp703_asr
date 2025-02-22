#!/bin/bash

# 1. Download utt_spk_text.tsv
FILE_URL_UTT="https://www.openslr.org/resources/35/utt_spk_text.tsv"
OUTPUT_FILENAME_UTT="utt_spk_text.tsv"

wget -O "$OUTPUT_FILENAME_UTT" "$FILE_URL_UTT"

if [ $? -eq 0 ]; then
  echo "File '$OUTPUT_FILENAME_UTT' downloaded successfully."
else
  echo "Error: Failed to download file from '$FILE_URL_UTT'."
  exit 1
fi


# 2. Download and Unpack asr_sundanese files

# Base URL of the zip files
BASE_URL="https://www.openslr.org/resources/35/asr_javanese_"
FILE_EXTENSION=".zip"

# Output directory for the ZIP files (create it if it doesn't exist)
OUTPUT_DIR="javanese_zips"
mkdir -p "$OUTPUT_DIR"

# Output directory for the unpacked data
UNPACKED_DIR="javanese_data"
mkdir -p "$UNPACKED_DIR"


# Loop through the zip files (0 to f)
# Use brace expansion with explicit numbers and letters
for i in {0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f}; do
  # FILE_NUMBER=$(printf "%x" $i) # No longer needed

  FILE_URL="$BASE_URL${i}${FILE_EXTENSION}"
  OUTPUT_FILENAME="$OUTPUT_DIR/asr_javanese_${i}${FILE_EXTENSION}"

  # Check if the file already exists before downloading
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

echo "All files downloaded and unpacked successfully."
exit 0
