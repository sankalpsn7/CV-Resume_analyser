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

async def chain3_match_and_score(hr_input: str, cv_data: dict) -> dict:
    """Asynchronously evaluates parsed CV data against HR requirements."""
    
    # --- ERROR CASCADING PROTECTION ---
    # If Chain 2 failed to read the PDF, do not waste API tokens trying to score it.
    if "parsing_error" in cv_data:
        return {
            "requirements": [],
            "score": 0.0,
            "summary": f"Could not evaluate candidate. Reason: {cv_data.get('parsing_error')}"
        }

    client = get_async_hf_client()
    
    prompt = f"""Compare the HR requirements against the candidate's CV data.

HR Requirements:
\"\"\"{hr_input}\"\"\"

CV Data:
{json.dumps(cv_data, indent=2, ensure_ascii=False)}

Return ONLY this precise JSON structure:
{{
  "requirements": [
    {{
      "criterion": "Short description of the requirement",
      "status": "✅ Met" or "❌ Not Found" or "⚠️ Partial",
      "evidence": "Exact text from CV or null"
    }}
  ],
  "score": <Float between 0.0 and 5.0>,
  "summary": "2-3 sentence executive assessment of the candidate's fit."
}}

JSON Output ONLY:"""

    # 'await' the network call
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a strict technical recruiter. Your ONLY output is a valid JSON object."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        max_tokens=1500, 
        temperature=0.1
    )
    
    try:
        # Rely on json_repair instead of brittle regex and remove non-English errors
        parsed_data = json.loads(repair_json(response.choices[0].message.content))
        
        if not isinstance(parsed_data, dict):
            raise ValueError("LLM did not return a JSON object.")
            
        return parsed_data
    except Exception as e:
        raise ValueError(f"Failed to parse LLM scoring output into JSON: {e}")