import streamlit as st
import torch
import pickle
import os
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import fitz  # PyMuPDF for PDF processing
import numpy as np
import re
from PIL import Image
from io import BytesIO

# Load DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", ignore_mismatched_sizes=True)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", ignore_mismatched_sizes=True)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Replace API keys and sensitive data with placeholders
API_KEY = ""  # Replace with your Azure OpenAI API key
RESOURCE_NAME = ""  # Replace with your Azure resource name
DEPLOYMENT_NAME = ""  # Replace with your deployment name
API_VERSION = ""  # Replace with your API version

BING_API_KEY = ""  # Replace with your Bing API key

# Function to clean and process extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'[^a-zA-Z0-9.,;:?!\-\(\)\s]', '', text)  # Remove special characters
    text = text.strip()  # Trim whitespace from start and end
    return text

# Function to load saved chunks, embeddings, and metadata
def load_embeddings_and_chunks(folder_path):
    with open(os.path.join(folder_path, 'chunks.pkl'), 'rb') as f:
        chunks = pickle.load(f)
    embeddings = torch.load(os.path.join(folder_path, 'context_embeddings.pt'), map_location=torch.device('cpu'))
    with open(os.path.join(folder_path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # Ensure embeddings are 2D
    if len(embeddings.shape) == 3:
        embeddings = embeddings.squeeze()
    elif len(embeddings.shape) != 2:
        raise ValueError(f"Expected embeddings to have 2 dimensions, but got shape: {embeddings.shape}")

    return chunks, embeddings, metadata

# Function to encode a user query using the DPR question encoder
def encode_question(question):
    inputs = question_tokenizer(question, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output.squeeze()
        if len(question_embedding.shape) != 1:
            raise ValueError(f"Expected question embedding to have 1 dimension, but got shape: {question_embedding.shape}")
    return question_embedding

# Function to retrieve the most relevant chunk using DPR embeddings
def retrieve_relevant_chunk_dpr(question, chunks, context_embeddings, metadata):
    question_embedding = encode_question(question)
    question_embedding = question_embedding.unsqueeze(0)  # Ensure it is 2D for cosine similarity
    context_embeddings = context_embeddings.cpu().numpy() if isinstance(context_embeddings, torch.Tensor) else context_embeddings

    scores = cosine_similarity(question_embedding, context_embeddings).flatten()
    best_chunk_index = scores.argmax()
    best_chunk = chunks[best_chunk_index]
    best_metadata = metadata[best_chunk_index]
    return best_chunk, best_metadata

# Function to refine the answer using Azure OpenAI GPT
def refine_answer_with_azure_openai(user_question, document_context):
    ENDPOINT = f"https://{RESOURCE_NAME}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant."},
            {"role": "system", "content": f"The following information is available from the documents:\n\n{document_context}"},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        completion = response.json()
        return completion['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        return f"Error with Azure OpenAI API: {e}"

# Function to encode text chunks into embeddings
def encode_context_batch(contexts):
    inputs = context_tokenizer(contexts, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        return context_encoder(**inputs).pooler_output

# Streamlit app main function
def main():
    # Create UI layout
    col1, col2 = st.columns([0.8, 4])
    logo_path = ""  # Replace with the path to your logo
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as img_file:
                img = Image.open(BytesIO(img_file.read()))
            with col1:
                st.image(img, width=80)
        except Exception as e:
            with col1:
                st.error(f"Error loading image: {e}")
    else:
        with col1:
            st.error("Logo file not found. Please check the path.")

    with col2:
        st.markdown("<h1 style='display: inline-block; vertical-align: middle;'>GAIA</h1>", unsafe_allow_html=True)
        st.write("I am your Grid AI Assistant! How can I help you?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if user_question := st.chat_input("Ask your question:"):
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Load data and retrieve answer
        folder_path = ""  # Replace with path to your data
        dev_chunks, dev_embeddings, dev_metadata = load_embeddings_and_chunks(folder_path)

        best_chunk, best_metadata = retrieve_relevant_chunk_dpr(user_question, dev_chunks, dev_embeddings, dev_metadata)

        if best_chunk:
            refined_answer = refine_answer_with_azure_openai(user_question, best_chunk)
            st.chat_message("assistant").markdown(refined_answer)
            st.session_state.messages.append({"role": "assistant", "content": refined_answer})

if __name__ == "__main__":
    main()
