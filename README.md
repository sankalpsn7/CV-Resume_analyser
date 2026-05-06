# 🤖 CV Analyzer — HR Match System

> Automatically analyze and rank multiple CVs against HR requirements using LLM-powered chains.

---

## 📽️ Demo


[![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/_zdc-hohSzs?si=qeGYnFZHJULQ7xZ_)


---

## ✨ Features

- 📋 **Chain 1** — Parses unstructured HR messages and extracts relevant CV fields automatically
- 📄 **Chain 2** — Extracts text from PDF CVs and maps content to each required field
- ⚖️ **Chain 3** — Scores each CV against HR requirements and generates a match report
- 📊 **Multi-CV Ranking** — Upload multiple CVs at once and get them ranked by score
- 🧾 **Detailed Table** — Per-criterion breakdown with evidence from the CV
- 🌐 **Streamlit UI** — Clean, fast web interface

---

## 🏗️ Project Structure

```
cv-analyzer/
│
├── .env                        ← API tokens 
├── .gitignore
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml             ← UI theme config
│
├── chains/
│   ├── __init__.py
│   ├── chain1.py               ← HR Input → Fields List
│   ├── chain2.py               ← PDF → Extracted CV Data
│   └── chain3.py               ← CV Data + HR → Score & Report
│
├── app.py                      ← Streamlit App
│
└── assets/
    └── demo.gif                ← Demo GIF 
```

---

## ⚙️ How It Works

```
HR Message (plain text)
        │
        ▼
   [ Chain 1 ]  →  Extracts required fields list
        │
        ▼
   PDF Upload
        │
        ▼
   [ Chain 2 ]  →  Maps CV content to each field
        │
        ▼
   [ Chain 3 ]  →  Scores & ranks CVs vs HR requirements
        │
        ▼
   📊 Ranked Table + Detailed Report
```

---

## 🚀 Getting Started

### 1 — Clone the repo
```bash
git clone https://github.com/Ahm495/cv-analyzer.git
cd cv-analyzer
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Set up your `.env` file
```bash
# .env
HF_TOKEN=your_huggingface_token_here
```
> Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4 — Run the app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit
huggingface-hub
pypdf
python-dotenv
pandas
json-repair
```

---

## 🧠 Model Used

| Component | Model |
|-----------|-------|
| All Chains | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API |

