import os
import json
import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional

# --- ENHANCED STRUCTURED OUTPUTS ---
class RequirementStatus(BaseModel):
    criterion: str = Field(description="Short description of the requirement")
    status: str = Field(description="Must be exactly '✅ Met', '❌ Not Found', or '⚠️ Partial'")
    evidence: str = Field(description="Exact text from CV or 'null'")
    # UPDATED: Now expects a list for multiple questions
    verification_questions: List[str] = Field(description="3-4 deep-dive technical questions to verify this specific claim.")

class CVScore(BaseModel):
    requirements: List[RequirementStatus]
    score: float 
    summary: str 
    red_flags: List[str]

def get_gemini_model():
    """Returns the latest stable Gemini 2.5 Flash model."""
    token = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not token:
        raise ValueError("GEMINI_API_KEY not found.")
    
    genai.configure(api_key=token)
    
    # Update this string to match the latest flash model in your list
    return genai.GenerativeModel('gemini-2.5-flash')

async def chain3_match_and_score(hr_input: str, cv_data: dict) -> dict:
    """Evaluates CV data and generates strategic interview insights using Pydantic schemas."""
    if "parsing_error" in cv_data:
        return {
            "requirements": [],
            "score": 0.0,
            "summary": f"Evaluation aborted. Reason: {cv_data.get('parsing_error')}",
            "red_flags": ["Could not parse CV"]
        }

    model = get_gemini_model()
    
    # Updated prompt to include the new logic
    prompt = f"""Compare the HR requirements against the candidate's CV data.
    
    1. For every requirement, provide status and evidence.
    2. Generate 3 to 4 varied 'verification_questions' for EACH requirement. Include:
       - One technical 'how-to' question.
       - One situational 'tell me about a time' question.
       - One deep-dive question into the evidence provided in the CV.
    3. Identify any 'red_flags' such as gaps or missing certifications.
    
    HR Requirements: {hr_input}
    CV Data: {json.dumps(cv_data, indent=2, ensure_ascii=False)}
    """

    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=CVScore, 
            temperature=0.0
        )
    )
    return json.loads(response.text)