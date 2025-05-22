#!/bin/bash

# Script to find LLM models on a Mac
# Created: April 27, 2025

echo "===== LLM Model Finder ====="
echo "Searching for machine learning models on your Mac..."
echo ""

# Define common locations to search
LOCATIONS=(
  "$HOME/.cache/huggingface"
  "$HOME/Library/Caches/huggingface"
  "$HOME/.huggingface"
  "$HOME/Library/Application Support/huggingface"
  "$HOME/Library/Application Support/torch"
  "$HOME/Library/Application Support/tensorflow"
  "$HOME/.keras"
  "$HOME/.ollama"
  "$HOME/Library/Application Support/ollama"
  "$HOME/.cache/torch"
  "$HOME/.cache/transformers"
  "$HOME/Library/Caches/transformers"
  "$HOME/Library/Containers"
)

# Define file patterns for LLMs
PATTERNS=(
  "*.gguf"
  "*.bin"
  "*.pt"
  "*.pth"
  "*.safetensors"
  "*.onnx"
  "*.pb"
  "model.json"
  "config.json"
  "pytorch_model.bin"
  "tokenizer.json"
)

# Function to format file size in human-readable format
format_size() {
    local size=$1
    if [ $size -ge 1073741824 ]; then
        echo "$(bc -l <<< "scale=2; $size/1073741824") GB"
    elif [ $size -ge 1048576 ]; then
        echo "$(bc -l <<< "scale=2; $size/1048576") MB"
    elif [ $size -ge 1024 ]; then
        echo "$(bc -l <<< "scale=1; $size/1024") KB"
    else
        echo "$size bytes"
    fi
}

# Create temporary file for results
TEMP_FILE=$(mktemp)

echo "Scanning system for LLM models..."
echo "This might take some time depending on your disk size..."
echo ""

# Search each location for the patterns
for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "Searching in $location"

        for pattern in "${PATTERNS[@]}"; do
            find "$location" -type f -name "$pattern" -not -path "*/\.*" 2>/dev/null | while read -r file; do
                size=$(stat -f "%z" "$file")
                if [ $size -gt 10485760 ]; then  # Only show files larger than 10MB
                    human_size=$(format_size $size)
                    echo "$file|$size|$human_size" >> "$TEMP_FILE"
                fi
            done
        done
    fi
done

# Also perform a broader search for very large model files across user directories
echo "Performing broader search for large model files..."
find "$HOME" -type f \( -name "*.gguf" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.bin" \) -size +100M 2>/dev/null | grep -v "node_modules" | while read -r file; do
    size=$(stat -f "%z" "$file")
    human_size=$(format_size $size)
    echo "$file|$size|$human_size" >> "$TEMP_FILE"
done

# Sort results by size (largest first)
if [ -s "$TEMP_FILE" ]; then
    echo ""
    echo "Found LLM model files (sorted by size):"
    echo "---------------------------------------"
    echo "Size | Location"
    echo "---------------------------------------"
    sort -t'|' -k2 -nr "$TEMP_FILE" | while IFS="|" read -r file size human_size; do
        echo "$human_size | $file"
    done

    # Calculate total size
    total_size=$(awk -F'|' '{sum+=$2} END {print sum}' "$TEMP_FILE")
    total_human_size=$(format_size $total_size)

    echo "---------------------------------------"
    echo "Total space used: $total_human_size"
    echo ""
    echo "To delete files, review the list carefully and use:"
    echo "rm /path/to/file"
    echo ""
    echo "For entire directories, use:"
    echo "rm -rf /path/to/directory"
    echo ""
    echo "WARNING: Be extremely careful with rm commands, especially with -rf flag!"
else
    echo "No large LLM model files found."
fi

# Clean up temp file
rm "$TEMP_FILE"

echo "Scan complete!"