# CLAUDE.md — Project Guidelines & Preferences

## Project Overview
Clinical data extraction pipeline using AI agents (smolagents + SNOMED CT).
3-agent pipeline: extractor → ontology mapper → reviewer.
Repo: https://github.com/rtdatasci/clinical_data_ai

## Tech Stack
- **Framework**: smolagents (HuggingFace)
- **LLM**: HuggingFace Inference API (free tier)
- **Data**: Synthetic clinical notes (JSON)
- **Ontology**: SNOMED CT (public browser API)
- **Environment**: Python, Windows 10, bash shell

## Project Structure
- `data/` — synthetic clinical notes
- `agents/` — extractor, ontology_mapper, reviewer agents
- `evaluate/` — ground truth + validator (precision/recall/F1)
- `results/` — generated at runtime (gitignored)
- `run_pipeline.py` — main orchestrator
- `.env` — HF_TOKEN (gitignored, never commit)

## Git & Commit Preferences
- Do NOT include "Co-Authored-By: Claude" in commit messages
- Keep commit messages concise and descriptive
- Always ensure `.env` and `results/` stay in `.gitignore`

## Coding Preferences
- Use `python-dotenv` for environment variable management
- Keep agents modular — one file per agent
- Parse agent outputs defensively (handle JSON extraction from LLM responses)
- Normalize clinical terms to lowercase for comparison

## Lessons Learned
- smolagents `CodeAgent` can return raw strings or dicts — always handle both
- SNOMED CT browser API (`browser.ihtsdotools.org`) is public, no auth needed
- Ground truth matching uses set-based comparison (normalize before comparing)
