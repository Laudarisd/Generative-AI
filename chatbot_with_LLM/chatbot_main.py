# Install necessary libraries
# !pip install -q transformers einops accelerate langchain bitsandbytes datasets

import os
import textwrap
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline as transformers_pipeline
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# Set environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_device_map(gpu_indices):
    if not gpu_indices:
        return {"": "cpu"}
    
    n_gpu = len(gpu_indices)
    if n_gpu == 1:
        return {"": f"cuda:{gpu_indices[0]}"}
    
    max_memory = {i: "16GiB" for i in gpu_indices}
    return infer_auto_device_map(model, max_memory=max_memory)

print("All the libraries are successfully imported")

# User-defined GPU indices
gpu_indices = [0, 1]  # Specify the GPUs you want to use, e.g., [0, 1, 2]

# Clear CUDA cache
torch.cuda.empty_cache()

# Check if GPUs are available
if torch.cuda.is_available():
    print(f"Using GPUs: {gpu_indices}")
else:
    gpu_indices = []
    print("No GPUs available, using CPU")

# Local directory where the model and tokenizer are saved
local_dir = "./huggyllama_llama_7b"

# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# Initialize an empty model and load the checkpoint directly onto the specified devices
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16)

# Create a device map for the specified GPUs
device_map = setup_device_map(gpu_indices)

# Load the model with the device map and offload to manage memory
model = load_checkpoint_and_dispatch(model, local_dir, device_map=device_map, offload_folder="offload")

pipeline = transformers_pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=device_map,
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True,
)

# LLM initialized in HuggingFacePipeline wrapper
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

# Load documents locally as CSV
csv_path = './CRM_data.csv'  # Update this with your local path
loader = CSVLoader(csv_path)
docs = loader.load()

# Split document into text chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs)

# Convert documents to a format suitable for RAG
qa_pairs = [{"question": doc.page_content.split("Answer:")[0].strip(), "answer": doc.page_content.split("Answer:")[1].strip()} for doc in docs]

# Design Multiple Prompt Templates
prompts = [
    PromptTemplate(template="""You are a helpful customer service chatbot for an online perfume company called Fragrances International.

Answer the customer's question clearly and concisely.

Context:
{context}

Question: {question}

Answer: """, input_variables=["context", "question"]),
    PromptTemplate(template="""You are a knowledgeable assistant for Fragrances International, an online perfume store.

Provide a helpful and detailed answer to the customer's question.

Context:
{context}

Question: {question}

Answer: """, input_variables=["context", "question"])
]

# Define a post-processing function for the responses
def post_process_response(response):
    lines = response.split('\n')
    processed_lines = [line.strip() for line in lines if line.strip()]
    processed_lines = ['. '.join([sentence.capitalize() for sentence in line.split('. ')]) for line in processed_lines]
    unique_lines = []
    seen = set()
    for line in processed_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    processed_response = ' '.join(unique_lines)
    return processed_response

# Example function to query the LLM with RAG and post-process the response
def query_llm(query: str, context: str, prompt_index: int = 0) -> str:
    full_prompt = prompts[prompt_index].format(context=context, question=query)
    raw_response = llm(full_prompt)
    processed_response = post_process_response(raw_response)
    return processed_response

# Interactive querying
def interactive_query():
    context = "\n".join([f"Question: {pair['question']}\nAnswer: {pair['answer']}" for pair in qa_pairs])
    while True:
        query = input("\nType your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        prompt_index = 0  # You can add logic to switch between different prompts if needed
        response = query_llm(query, context, prompt_index)
        print_response(response)

# Fine-Tune the Model
def fine_tune_model(dataset_path: str):
    dataset = load_dataset('csv', data_files=dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples['Question'] + " " + examples['Answer'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=2,  # Reduced batch size
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
    )

    trainer.train()

# Function to print the response
def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=80)))

# Optionally fine-tune the model
# fine_tune_model('./extended_CRM_data.csv')

# Start the interactive querying
interactive_query()
