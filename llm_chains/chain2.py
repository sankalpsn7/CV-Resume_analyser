import os
import json
import streamlit as st
from pypdf import PdfReader
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

def extract_cv_text(pdf_file) -> str:
    """Safely extracts text from a PDF, handling corrupt or unreadable files."""
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        
        if not text.strip():
            return "ERROR: PDF appears to be empty or consists entirely of unreadable scanned images."
            
        return text.strip()
    except Exception as e:
        return f"ERROR: Failed to read PDF format. Details: {e}"

async def chain2_parse_cv(cv_text: str, fields_list: list) -> dict:
    """Asynchronously parses CV text against extracted fields into structured JSON."""
    
    # Early exit mechanism if the PDF was unreadable
    if cv_text.startswith("ERROR:"):
        return {"parsing_error": cv_text}

    client = get_async_hf_client()
    
    prompt = f"""Extract information from the CV for each required field below.
- Return ONLY a JSON object.
- Use null if a field is not found.
- Use arrays for list-type fields.

Required Fields: {fields_list}

CV Text:
\"\"\"{cv_text}\"\"\"

JSON Output ONLY:"""

    # 'await' the network call so the thread can process the next CV concurrently
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a professional CV parser. Your ONLY output is a valid JSON object."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        max_tokens=3000, 
        temperature=0.1
    )
    
    try:
        # json_repair eliminates the need for brittle Regular Expressions
        parsed_data = json.loads(repair_json(response.choices[0].message.content))
        
        if not isinstance(parsed_data, dict):
            raise ValueError("LLM did not return a dictionary object.")
            
        return parsed_data
    except Exception as e:
        raise ValueError(f"Failed to parse CV extraction into JSON: {e}")