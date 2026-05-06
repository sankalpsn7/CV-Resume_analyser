import streamlit as st
import pandas as pd
from llm_chains.chain1 import chain1_extract_fields
from llm_chains.chain2 import extract_cv_text, chain2_parse_cv
from llm_chains.chain3 import chain3_match_and_score

st.set_page_config(page_title="CV Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered CV Analyzer")
st.markdown("Extract core requirements from an HR prompt and dynamically score candidate CVs.")
st.divider()

# ── Step 1: HR Input ─────────────────────────────────────
st.subheader("1️⃣ HR Requirements")
hr_input = st.text_area("Enter the HR message or Job Description", height=150, label_visibility="collapsed")

if st.button("🔍 Extract Requirements", type="primary", use_container_width=True):
    if not hr_input.strip():
        st.warning("⚠️ Please enter the HR requirements first.")
    else:
        with st.spinner("⏳ Analyzing requirements via LLM..."):
            try:
                fields = chain1_extract_fields(hr_input)
                st.session_state["fields"]   = fields
                st.session_state["hr_input"] = hr_input
                st.session_state.pop("results", None)  # Clear previous run
            except Exception as e:
                st.error(f"❌ Failed to extract fields: {e}")

# ── Step 2: Upload & Process ─────────────────────────────
if "fields" in st.session_state:
    st.divider()
    st.subheader("2️⃣ Extracted Extraction Targets")
    cols = st.columns(3)
    for i, f in enumerate(st.session_state["fields"]):
        cols[i % 3].success(f"✅ {f}")

    st.divider()
    st.subheader("3️⃣ Batch Upload CVs")
    cv_files = st.file_uploader(
        "Upload candidate resumes (PDF format)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if cv_files and st.button("🚀 Execute Analysis Pipeline", type="primary", use_container_width=True):
        results = []
        progress_bar = st.progress(0.0, text="Initializing pipeline...")
        
        for idx, cv_file in enumerate(cv_files):
            status_text = f"⏳ Processing: {cv_file.name} ({idx+1}/{len(cv_files)})"
            progress_bar.progress(idx / len(cv_files), text=status_text)
            
            with st.spinner(status_text):
                try:
                    cv_text = extract_cv_text(cv_file)
                    cv_data = chain2_parse_cv(cv_text, st.session_state["fields"])
                    result  = chain3_match_and_score(st.session_state["hr_input"], cv_data)
                    
                    results.append({
                        "filename": cv_file.name,
                        "result":   result
                    })
                except Exception as e:
                    st.error(f"❌ Error processing {cv_file.name}: {e}")
                    
        progress_bar.progress(1.0, text="✅ Pipeline execution complete!")
        st.session_state["results"] = results

# ── Step 3: Results & Ranking ────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    
    st.divider()
    st.subheader("📊 Candidate Ranking Matrix")

    ranking_data = []
    # Safely handle the score sorting even if LLM missed it
    sorted_results = sorted(
        results, 
        key=lambda x: float(x["result"].get("score", 0) or 0), 
        reverse=True
    )

    for rank, r in enumerate(sorted_results, 1):
        # Bulletproof type casting for LLM numeric outputs
        raw_score = r["result"].get("score", 0)
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            score = 0.0
            
        score = max(0.0, min(score, 5.0)) # Clamp between 0 and 5
        filled = int(round(score))
        bar = "█" * filled + "░" * (5 - filled)
        
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f"{rank}"
        
        ranking_data.append({
            "Rank": medal,
            "Candidate File": r["filename"],
            "Match Score": f"{bar}  {score:.1f} / 5.0",
            "LLM Summary": r["result"].get("summary", "No summary provided.")
        })
        
    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

    # ── Detailed CV Breakdown ────────────────────────────
    st.divider()
    st.subheader("🔍 Extraction Details")

    tabs = st.tabs([f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '📄'} {r['filename']}" 
                    for i, r in enumerate(sorted_results)])

    for tab, r in zip(tabs, sorted_results):
        with tab:
            result  = r["result"]
            reqs    = result.get("requirements", [])
            
            try:
                score = float(result.get("score", 0))
            except (ValueError, TypeError):
                score = 0.0
            score = max(0.0, min(score, 5.0))

            # Render Details Table without truncation
            df = pd.DataFrame([{
                "Criterion": req.get("criterion", "N/A"),
                "Status": req.get("status", "Unknown"),
                "Extracted Evidence": req.get("evidence", "No evidence found.") 
            } for req in reqs])
            
            st.dataframe(df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("🏆 Calculated Score", f"{score:.1f} / 5.0")
            with col2:
                st.progress(score / 5.0)

            st.divider()
            st.markdown("**Executive Summary:**")
            st.write(result.get("summary", "N/A"))