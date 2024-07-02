"""
This code downloads the HuggyLLama/llama-7b model and tokenizer from the Hugging Face Hub.
The model and tokenizer are saved locally in a directory called "huggyllama_llama_7b".
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model name
model_name = "huggyllama/llama-7b"
# Local directory to save the model and tokenizer
local_dir = "./huggyllama_llama_7b"
os.makedirs(local_dir,  exist_ok=True)

# Download and save the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)