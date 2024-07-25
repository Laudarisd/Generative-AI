import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as transformers_pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders.csv_loader import CSVLoader

# Set environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@st.cache_resource
def setup_model():
    # Local directory where the model and tokenizer are saved
    local_dir = "./huggyllama_llama_7b"

    # Load the tokenizer from the local directory
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # Initialize an empty model and load the checkpoint directly onto the specified devices
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16)

    # Create a device map for the specified GPUs
    gpu_indices = [0, 1]  # Specify the GPUs you want to use, e.g., [0, 1, 2]
    if torch.cuda.is_available():
        print(f"Using GPUs: {gpu_indices}")
        n_gpu = len(gpu_indices)
        if n_gpu == 1:
            device_map = {"": f"cuda:{gpu_indices[0]}"}
        else:
            max_memory = {i: "16GiB" for i in gpu_indices}
            device_map = infer_auto_device_map(model, max_memory=max_memory)
    else:
        print("No GPUs available, using CPU")
        device_map = {"": "cpu"}

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
    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0.3})

    return llm

@st.cache_resource
def load_qa_pairs():
    # Load documents locally as CSV
    csv_path = './CRM_data.csv'  # Update this with your local path
    loader = CSVLoader(csv_path)
    docs = loader.load()

    # Convert documents to a format suitable for RAG
    qa_pairs = [{"question": doc.page_content.split("Answer:")[0].strip(), "answer": doc.page_content.split("Answer:")[1].strip()} for doc in docs]
    return qa_pairs

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_qa(query, qa_pairs, sentence_model, similarity_threshold=0.5):
    query_embedding = sentence_model.encode([query])
    question_embeddings = sentence_model.encode([pair['question'] for pair in qa_pairs])
    
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    most_similar_index = similarities.argmax()
    
    if similarities[most_similar_index] >= similarity_threshold:
        return qa_pairs[most_similar_index]
    else:
        return None

def post_process_response(response):
    # Clean up the response text
    clean_response = response.strip()

    # Capitalize the first letter of each sentence
    sentences = clean_response.split('. ')
    sentences = [sentence.capitalize() for sentence in sentences]

    # Join the sentences back into a single string
    processed_response = '. '.join(sentences)
    if not processed_response.endswith('.'):
        processed_response += '.'

    return processed_response

few_shot_template = """You are a helpful customer service chatbot for an online perfume company called Fragrances International. Answer the customer's question based on the given context. If the context doesn't contain relevant information, say you don't have enough information to answer accurately. Provide a single, concise answer without numbering or listing multiple points.

Context:
{context}

Customer Question: {question}

Answer: """

prompt = PromptTemplate(template=few_shot_template, input_variables=["context", "question"])

def query_llm(query: str, llm, qa_pairs: list, sentence_model) -> str:
    # Get the most relevant QA pair
    relevant_pair = get_most_relevant_qa(query, qa_pairs, sentence_model)
    
    if relevant_pair is None:
        return "I'm sorry, I don't have enough information to answer that question accurately. Could you please rephrase or ask something else?"

    context = f"Question: {relevant_pair['question']}\nAnswer: {relevant_pair['answer']}"
    full_prompt = prompt.format(context=context, question=query)
    
    raw_response = llm.invoke(full_prompt)
    
    # Extract the generated text
    try:
        if isinstance(raw_response, list) and len(raw_response) > 0:
            if isinstance(raw_response[0], dict) and 'generated_text' in raw_response[0]:
                generated_text = raw_response[0]['generated_text']
            else:
                generated_text = raw_response[0]
        else:
            generated_text = raw_response
    except Exception as e:
        print(f"Error extracting generated text: {e}")
        return "An error occurred processing the response."

    if not generated_text:
        return "An error occurred: no generated text found."

    # Extract the answer part from the generated text
    answer_start = generated_text.lower().find("answer:")
    if answer_start != -1:
        answer = generated_text[answer_start + 7:].strip()
        # Extract only the first sentence
        first_sentence_end = answer.find('.')
        if first_sentence_end != -1:
            answer = answer[:first_sentence_end + 1]
    else:
        # If "Answer:" is not found, take the first sentence of the generated text
        first_sentence_end = generated_text.find('.')
        if first_sentence_end != -1:
            answer = generated_text[:first_sentence_end + 1]
        else:
            answer = generated_text

    processed_response = post_process_response(answer)
    return processed_response

def main():
    st.title("Fragrances International Chatbot")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load resources
    llm = setup_model()
    qa_pairs = load_qa_pairs()
    sentence_model = load_sentence_model()

    # User input
    user_input = st.text_input("Ask a question about our perfumes:")

    if st.button("Send"):
        if user_input:
            # Get chatbot response
            response = query_llm(user_input, llm, qa_pairs, sentence_model)

            # Add to chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Chatbot", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Chatbot:** {message}")

if __name__ == "__main__":
    main()
