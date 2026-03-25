# AMA — Ask Me Anything | Sim Yew Chong

## Overview
An AI-powered personal chatbot that answers questions about Sim Yew Chong's professional background, experience, skills, and projects. Built with Streamlit, OpenAI, and FAISS-based RAG (Retrieval-Augmented Generation), featuring a sliding-window conversation history and defense-in-depth jailbreak protection.

## Architecture
```
User Input → Safety Check (is_safe_query)
                    ↓ PASS
            RAG Retriever (FAISS, top_k=2)
                    ↓
            OpenAI Chat (gpt-5.4)
            + Sliding Window History (last 3 turns)
                    ↓
            Response + Sources Shown in UI
```

## Setup
1. `pip install -r requirements.txt`
2. `cp .env.example .env` → add your `OPENAI_API_KEY`
3. `streamlit run app.py`
   (FAISS index auto-builds on first run from `resume/` folder)

## Running Evaluation
```bash
python evaluation/evaluate.py
```

## Design Decisions
- **Why FAISS** — Local, no external service dependency, sufficient for 6 chunks
- **Why top_k=2** — 6 chunks total; top 3 = 50% of KB = too noisy
- **Why sliding window [-6:]** — Last 3 turns balances context retention vs token cost
- **Why keyword-based safety filter** — Fast, zero latency, no extra API call needed
- **Why text-embedding-3-large** — Strong multilingual + English performance on MTEB benchmarks

## Knowledge Base Sources
- Resume (6 manually pre-chunked `.txt` files in `/resume`)
- Future: LinkedIn, GitHub, personal website scraper

## Known Limitations & Future Improvements
- Knowledge base is static (no live updates)
- Jailbreak filter is keyword-based only (can be improved with LLM-based classifier)
- No authentication (anyone with the URL can access)
- Future: dynamic ingestion pipeline for multiple document types
- Future: user-customizable profiles
