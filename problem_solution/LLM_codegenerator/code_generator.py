import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the code generation model and tokenizer
@st.cache_resource
def load_model(model_name="Salesforce/codegen-350M-mono"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Load the sentence transformer model for embedding
@st.cache_resource
def load_embedding_model(model_name="paraphrase-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

embedding_model = load_embedding_model()

# Improved RAG (Retrieval-Augmented Generation) setup
def retrieve_relevant_context(prompt, context_documents):
    prompt_embedding = embedding_model.encode([prompt])
    context_embeddings = embedding_model.encode(context_documents)
    similarities = cosine_similarity(prompt_embedding, context_embeddings)
    most_similar_idx = similarities.argmax()
    return context_documents[most_similar_idx]

def generate_code(prompt, tokenizer, model, max_length=1024, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Ensure the input sequence length is within bounds to avoid IndexError
    if inputs["input_ids"].shape[-1] > 1024:
        inputs["input_ids"] = inputs["input_ids"][:, :1024]

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Focused context documents related to algorithms and prime numbers
context_documents = [
    "A prime number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers...",
    "The Sieve of Eratosthenes is an efficient algorithm to find all prime numbers up to a given limit...",
    "To check if a number is prime, you can test for divisibility from 2 up to the square root of the number..."
]

st.title("Code Generation with Improved Context and Prompting")

# User input and context retrieval
user_input = st.text_area("Enter your code prompt here:")
temperature = st.slider("Select the temperature for code generation:", 0.1, 1.0, 0.7)

if st.button("Generate Code"):
    if user_input.strip():
        relevant_context = retrieve_relevant_context(user_input, context_documents)

        # Refine the prompt by including the retrieved context and a clearer task definition
        refined_prompt = (
            f"{relevant_context}\n\n"
            "Task: Write a Python function that finds all prime numbers in a given range. "
            "The function should take two arguments: the start and end of the range.\n\n"
            "Example:\n"
            "def find_primes_in_range(start, end):\n"
            "    # Your code here\n\n"
            f"{user_input}"
        )

        generated_code = generate_code(refined_prompt, tokenizer, model, temperature=temperature)
        st.code(generated_code, language="python")
    else:
        st.warning("Please enter a code prompt to generate code.")
