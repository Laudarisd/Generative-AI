"""
Chatbot with LLMS
- To run the program: streamlit run test-2.py

- Requirements : 
    1. streamlit
    2. transformers
    3. langchain
    4. Model in the local directory or huggingface token
    5. Data related to the model
"""
__Creatred_by__ = "Sudip Laudari"
__Date__ = "2024-07-26"

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
def setup_model(model_name_or_path: str = "huggingface/model_name") -> HuggingFacePipeline:    
    # Local directory where the model and tokenizer are saved
    local_dir = "./huggyllama_llama_7b"
    #check if the model exists in the local directory
    if os.path.exists(local_dir):
        model_dir = local_dir
    else:
        model_dir = model_name_or_path
        
    #The tokenizer is responsible for converting text into tokens that the model can understand and vice versa.

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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

    #A text-generation pipeline is created, specifying various parameters like max_length, do_sample, top_k, etc. 
    #This pipeline will handle the text generation process.
    
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
    
    #The temperature parameter is a crucial hyperparameter in the context of language models, 
    #particularly during the text generation process. 
    #It controls the randomness of the predictions made by the model
    #Low Temperature (< 1.0) = Less randomness
    #High Temperature (> 1.0) = More randomness
    
    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0.3})

    return llm

@st.cache_resource
def load_custom_data():
    # Load documents locally as CSV
    csv_path = './CRM_data.csv'  # Update this with your local path
    loader = CSVLoader(csv_path)
    docs = loader.load()

    # Convert documents to a format suitable for RAG
    
    #This part of the function processes the loaded documents to extract questions and answers from each document.
    #convert_data = [{"question": doc.page_content.split("Answer:")[0].strip(), 
    #"answer": doc.page_content.split("Answer:")[1].strip()} for doc in docs]:
    
    #This line uses a list comprehension to iterate over each document in docs.
    #doc.page_content.split("Answer:"): Splits the content of each document into two parts at the substring "Answer:".
    #.strip(): Removes any leading and trailing whitespace from the split parts.
    #{"question": ..., "answer": ...}: Creates a dictionary with two keys, question and answer, holding the respective parts of the document content.
    #This transformation prepares the data in a format suitable for use in a Retrieval-Augmented Generation (RAG) system, 
    #where the question and answer pairs can be used to train or query the model.
    
    #- Data Format: By converting the documents into a dictionary format with question and answer keys, 
    #the function makes the data ready for further processing or model training.
    #
    convert_data = [{"question": doc.page_content.split("Answer:")[0].strip(), "answer": doc.page_content.split("Answer:")[1].strip()} for doc in docs]
    return convert_data

@st.cache_resource
def load_sentence_model():
    """
    SentenceTransformer('all-MiniLM-L6-v2'): Loads the pre-trained SentenceTransformer model with the identifier 'all-MiniLM-L6-v2'.
    The SentenceTransformer is a popular model from the sentence-transformers library, which is used for tasks like semantic textual similarity, clustering, and more.
    The 'all-MiniLM-L6-v2' model is a lightweight, efficient model that provides good performance for a variety of tasks involving sentence embeddings.
    This returns the loaded SentenceTransformer model. The returned model can then be used for various NLP tasks such as generating embeddings for sentences or documents.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_qa(query, convert_data, sentence_model, similarity_threshold=0.5): 
    query_embedding = sentence_model.encode([query]) # The query is encoded into an embedding using the sentence_model. The resulting query_embedding is a numerical representation of the query in a high-dimensional space.
    question_embeddings = sentence_model.encode([pair['question'] for pair in convert_data])#This encodes all the questions from convert_data into embeddings
    
    similarities = cosine_similarity(query_embedding, question_embeddings)[0] #The cosine similarity between the query embedding and each question embedding is calculated.
    most_similar_index = similarities.argmax()
    
    if similarities[most_similar_index] >= similarity_threshold: # If the similarity score is above the threshold, the most relevant question-answer pair is returned.
        return convert_data[most_similar_index]
    else:
        return None

def post_process_response(response):
    """
    Clean and format the response text.
    """
    # Clean up the response text
    clean_response = response.strip() # Remove leading and trailing whitespace

    # Capitalize the first letter of each sentence
    sentences = clean_response.split('. ')  # Split the response into list of sentences
    sentences = [sentence.capitalize() for sentence in sentences] # Iterates over each sentence and capitalizes the first letter, ensuring proper sentence capitalization.

    # Join the sentences back into a single string
    processed_response = '. '.join(sentences) #Joins the list of sentences back into a single string, with each sentence separated by a period and a space.
    if not processed_response.endswith('.'): # processed_response += '.': Ensures that the final response ends with a period, adding one if necessary.

        processed_response += '.'

    return processed_response


#The few_shot_template and PromptTemplate are used to define a structured prompt for the language model, helping it generate responses in a specific context

few_shot_template = """You are a helpful customer service chatbot for an online perfume company called Fragrances International. Answer the customer's question based on the given context. If the context doesn't contain relevant information, say you don't have enough information to answer accurately. Provide a single, concise answer without numbering or listing multiple points.

Context:
{context}

Customer Question: {question}

Answer: """

prompt = PromptTemplate(template=few_shot_template, input_variables=["context", "question"])

def query_llm(query: str, llm, convert_data: list, sentence_model) -> str:
    # Get the most relevant QA pair
    relevant_pair = get_most_relevant_qa(query, convert_data, sentence_model)
    
    if relevant_pair is None:
        return "I'm sorry, I don't have enough information to answer that question accurately. Could you please rephrase or ask something else?"

    # The language model is queried with the full prompt, which includes both the 
    # context from our data set and the user's question.


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
    st.title("Welcome to the Fragrances International AI Assistant Service")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load resources
    llm = setup_model()
    convert_data = load_custom_data()
    sentence_model = load_sentence_model()

    # User input
    user_input = st.text_input("Ask a question about our products:")

    if st.button("Send"):
        if user_input:
            # Get chatbot response
            response = query_llm(user_input, llm, convert_data, sentence_model)

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
