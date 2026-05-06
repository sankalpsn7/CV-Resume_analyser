import os
import json
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

def get_gemini_model():
    """Returns the latest stable Gemini 2.5 Flash model."""
    token = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not token:
        raise ValueError("GEMINI_API_KEY not found.")
    
    genai.configure(api_key=token)
    
    # Update this string to match the latest flash model in your list
    return genai.GenerativeModel('gemini-2.5-flash')

def extract_cv_text(pdf_file) -> str:
    """Safely extracts text from a PDF, handling corrupt files."""
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return "ERROR: PDF appears empty or contains only images."
        return text.strip()
    except Exception as e:
        return f"ERROR: Failed to read PDF format. Details: {e}"

async def chain2_parse_cv(cv_text: str, fields_list: list) -> dict:
    """Parses CV text against extracted fields into structured JSON."""
    if cv_text.startswith("ERROR:"):
        return {"parsing_error": cv_text}

    model = get_gemini_model()
    prompt = f"""Extract information from the CV for each required field below.
    Required Fields: {fields_list}
    
    CV Text:
    {cv_text}
    """
    
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    
    return json.loads(response.text)