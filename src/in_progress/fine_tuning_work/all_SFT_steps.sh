#!/bin/bash
# run script to fine-tune models
python3 SFT.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "script_a.py failed!"
    exit 1
fi

# Run script to perform certs with fine-tuned .py file
python3 SFT_CeRTS.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "SFT_CeRTS.py failed!"
    exit 1
fi

# Run script to perform evaluation of CeRTS results
python3 SFT_CeRTS_eval.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "SFT_CeRTS_eval.py failed!"
    exit 1
fi

echo "All scripts ran successfully!"