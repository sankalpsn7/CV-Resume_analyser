import os
import json
import streamlit as st
import google.generativeai as genai
def get_gemini_model():
    """Returns the latest stable Gemini 2.5 Flash model."""
    token = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not token:
        raise ValueError("GEMINI_API_KEY not found.")
    
    genai.configure(api_key=token)
    
    # Update this string to match the latest flash model in your list
    return genai.GenerativeModel('gemini-2.5-flash')

async def chain1_extract_fields(hr_input: str) -> list:
    """Asynchronously extracts 3-12 required fields from HR input for CV screening."""
    model = get_gemini_model()
    
    prompt = f"""You are an expert HR analyst. Analyze the job requirements below and 
    extract a list of 3 to 12 specific fields or criteria to verify in a candidate's CV.
    
    Return ONLY a JSON array of strings.
    
    Requirements:
    \"\"\"{hr_input}\"\"\"
    
    JSON Array:"""
    
    try:
        # Utilizing generate_content_async for non-blocking I/O
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json", # Forces JSON output natively
                temperature=0.1
            )
        )
        
        # Parse the JSON string directly into a Python list
        fields = json.loads(response.text)
        
        if not isinstance(fields, list):
            raise ValueError("LLM output is not a valid list.")
            
        return fields
        
    except Exception as e:
        # Provides clear error feedback in the Streamlit UI
        raise ValueError(f"Chain 1 Error: {str(e)}")

