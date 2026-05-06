import streamlit as st
import pandas as pd
import asyncio
from llm_chains.chain1 import chain1_extract_fields
from llm_chains.chain2 import extract_cv_text, chain2_parse_cv
from llm_chains.chain3 import chain3_match_and_score

st.set_page_config(page_title="AI CV Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered CV Analyzer")
st.markdown("Extract HR requirements and dynamically rank candidates using Gemini Flash.")
st.divider()

# --- Async Helper Function for Batch Processing ---
async def process_cv_pipeline(cv_file, hr_input, fields):
    """Handles the async chain2 -> chain3 pipeline for a single CV."""
    try:
        # PDF reading is sync, so we do it first
        cv_text = extract_cv_text(cv_file)
        
        # Async LLM calls
        cv_data = await chain2_parse_cv(cv_text, fields)
        result = await chain3_match_and_score(hr_input, cv_data)
        
        return {"filename": cv_file.name, "result": result}
    except Exception as e:
        return {"filename": cv_file.name, "error": str(e)}

async def run_batch_analysis(cv_files, hr_input, fields):
    """Fires all CV pipelines concurrently."""
    tasks = [process_cv_pipeline(cv, hr_input, fields) for cv in cv_files]
    return await asyncio.gather(*tasks)

# ── Step 1: HR Input ─────────────────────────────────────
st.subheader("1️⃣ HR Requirements")
hr_input = st.text_area("Enter Job Description", height=150, label_visibility="collapsed")

if st.button("🔍 Extract Requirements", type="primary", width='stretch'):
    if hr_input.strip():
        with st.spinner("⏳ Analyzing requirements via Gemini..."):
            try:
                # We use asyncio.run to execute the async function from sync Streamlit
                fields = asyncio.run(chain1_extract_fields(hr_input))
                st.session_state["fields"] = fields
                st.session_state["hr_input"] = hr_input
                st.session_state.pop("results", None)
            except Exception as e:
                st.error(f"❌ Failed to extract fields: {e}")

# ── Step 2: Upload & Process ─────────────────────────────
if "fields" in st.session_state:
    st.divider()
    cols = st.columns(3)
    for i, f in enumerate(st.session_state["fields"]):
        cols[i % 3].success(f"✅ {f}")

    st.divider()
    st.subheader("2️⃣ Batch Upload CVs")
    cv_files = st.file_uploader(
        "Upload candidate resumes (PDF format)", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if cv_files and st.button("🚀 Execute Concurrent Analysis", type="primary", width='stretch'):
        with st.spinner(f"⏳ Processing {len(cv_files)} CV(s) concurrently via Gemini 1.5 Flash..."):
            # Fire all async tasks at exactly the same time
            results = asyncio.run(run_batch_analysis(cv_files, st.session_state["hr_input"], st.session_state["fields"]))
            
            # Filter out any that crashed completely
            valid_results = [r for r in results if "error" not in r]
            errors = [r for r in results if "error" in r]
            
            for err in errors:
                st.error(f"Failed to process {err['filename']}: {err['error']}")
                
            st.session_state["results"] = valid_results

# ── Step 3: Results & Ranking ────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    st.divider()
    st.subheader("📊 Candidate Leaderboard")

    sorted_results = sorted(results, key=lambda x: float(x["result"].get("score", 0)), reverse=True)
    ranking_data = []

    # 1. Clean Leaderboard (No long text, just core metrics)
    for rank, r in enumerate(sorted_results, 1):
        score = max(0.0, min(float(r["result"].get("score", 0)), 5.0))
        red_flag_count = len(r["result"].get("red_flags", []))
        flag_icon = "🚩" if red_flag_count > 0 else "✅"
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f"{rank}"
        
        ranking_data.append({
            "Rank": medal,
            "Candidate": r["filename"],
            "Score": score,  # Kept as a raw float so we can use a native Progress Bar
            "Alerts": f"{flag_icon} {red_flag_count} Flags"
        })
        
    # HUGE FLEX: Using st.column_config to render a native progress bar inside the table
    st.dataframe(
        pd.DataFrame(ranking_data),
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
            "Score": st.column_config.ProgressColumn(
                "Match Score",
                help="AI calculated fit score out of 5.0",
                format="%.1f",
                min_value=0,
                max_value=5,
            ),
            "Alerts": st.column_config.TextColumn("Alerts", width="small"),
        },
        width='stretch',
        hide_index=True
    )

    # ── Detailed Interviewer Insights ────────────────────────────
    st.divider()
    st.subheader("🔍 Deep Dive & Interview Strategy")
    tabs = st.tabs([f"{'📄'} {r['filename']}" for r in sorted_results])

    for tab, r in zip(tabs, sorted_results):
        with tab:
            result = r["result"]
            score = max(0.0, min(float(result.get("score", 0)), 5.0))

            # 2. Top-level Summary & Metrics
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("🏆 Overall Fit Score", f"{score:.1f} / 5.0")
            with col2:
                st.info(f"**Executive Assessment:** {result.get('summary', 'N/A')}")

            # 3. Clean List for Red Flags
            red_flags = result.get("red_flags", [])
            if red_flags:
                st.error("#### 🚩 Critical Red Flags & Gaps")
                for flag in red_flags:
                    st.write(f"- {flag}")
            else:
                st.success("✅ No significant red flags or gaps detected in this CV.")

            st.divider()
            st.markdown("#### 📋 Skills Breakdown & Verification")
            # This allows us to show the exact evidence and the generated interview questions without truncation or formatting issues.
            for req in result.get("requirements", []):
                status = req.get("status", "Unknown")
                criterion = req.get("criterion", "N/A")
                icon = "✅" if "Met" in status else "❌" if "Not Found" in status else "⚠️"
                with st.expander(f"{icon} {criterion}"):
                    st.markdown(f"**Extracted Evidence:**\n> {req.get('evidence', 'No evidence found.')}")
                    st.markdown("**💡 Suggested Interview Questions:**")
                    questions = req.get("verification_questions", [])
                    if questions:
                        for q in questions:
                            st.write(f"- {q}")
                    else:
                        st.write("- No questions generated.")