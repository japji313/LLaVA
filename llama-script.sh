#!/bin/bash

# Set the maximum number of retries
max_retries=5

# Set the delay between retries (in seconds)
retry_delay=10

# Set the path to your Python script
python_script="python -m llava.serve.cli2 --model-path liuhaotian/llava-v1.6-34b"

# Initialize the retry counter
retry_count=0

# Function to run the Python script
run_script() {
    python "$python_script" --folder_path "images"
    return $?
}

# Loop until the script succeeds or the maximum retries are reached
while true; do
    run_script

    # Check the exit status of the Python script
    if [ $? -eq 0 ]; then
        echo "Python script executed successfully."
        break
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -le $max_retries ]; then
            echo "Python script failed. Retrying in $retry_delay seconds... (Retry $retry_count/$max_retries)"
            sleep $retry_delay
        else
            echo "Python script failed after $max_retries retries. Exiting."
            exit 1
        fi
    fi
done
