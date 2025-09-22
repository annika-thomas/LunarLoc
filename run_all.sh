#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 path/to/script.py"
    exit 1
fi
PYTHON_SCRIPT="$1"

for file in data/*; do
    filename=$(basename "$file")
    if ! python "$PYTHON_SCRIPT" -t "$filename" -s; then
        echo -e "\033[31mERROR WITH DATASET: $filename\033[0m"
    fi
done


# TODO: 
# - Look into IMU only odometry