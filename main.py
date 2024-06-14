import time
from huggingface_hub import InferenceClient
import json
import requests  # Import requests to handle HTTP errors
from dotenv import load_dotenv  # Import the load_dotenv function
import os
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Ensure the token is set correctly
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token is not set or invalid. Please set it in the .env file.")

repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
    token=hf_token
)

def call_llm(inference_client: InferenceClient, prompt: str, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = inference_client.post(
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 200},  # Limit the number of tokens
                    "task": "text-generation",
                },
            )
            generated_text = json.loads(response.decode())[0]["generated_text"]
            return generated_text.strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit reached. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    raise Exception("Failed to get a response after several retries")

# Refine the prompt to be more specific
prompt = "Tell me a single, crazy joke. Make it funny."
response = call_llm(llm_client, prompt)
st.write(response)