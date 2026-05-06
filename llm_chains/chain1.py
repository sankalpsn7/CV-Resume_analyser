import os
import json
import streamlit as st
from huggingface_hub import AsyncInferenceClient
from json_repair import repair_json

def get_async_hf_client() -> AsyncInferenceClient:
    """Safely fetch the token and return an asynchronous client."""
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            token = st.secrets["HF_TOKEN"]
        except (FileNotFoundError, KeyError):
            raise ValueError("HF_TOKEN not found in environment or Streamlit secrets.")
            
    return AsyncInferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)

async def chain1_extract_fields(hr_input: str) -> list:
    """Asynchronously extracts required fields from the HR input."""
    client = get_async_hf_client()
    
    # We MUST 'await' the network call so the thread isn't blocked
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are an expert HR analyst. Your ONLY output is a valid JSON array of strings."
            },
            {
                "role": "user", 
                "content": f"Extract 3-12 core CV evaluation fields based on these HR requirements.\n\nRequirements:\n{hr_input}\n\nJSON Array ONLY:"
            }
        ], 
        max_tokens=300, 
        temperature=0.1
    )
    
    try:
        # json_repair handles any markdown fences natively
        parsed_data = json.loads(repair_json(response.choices[0].message.content))
        if not isinstance(parsed_data, list):
            raise ValueError("LLM did not return a list.")
        return parsed_data
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output into JSON array: {e}")