# Clinical Data Extraction using AI Agents

A pipeline that reads clinical notes and automatically pulls out medical information — conditions, medications, procedures, and body parts — then maps them to standard medical codes (SNOMED CT) and generates a quality report.

## Why This Project?

Clinical notes are written in free text. To use that information for research or analytics, you need structured data. This project automates that extraction using AI agents that work together like a small team.

## Why smolagents?

[smolagents](https://github.com/huggingface/smolagents) is a lightweight agent framework from HuggingFace. It lets you build AI agents that can write and run code, use tools (like web search or API calls), and chain together in a pipeline — all with minimal setup. It connects directly to HuggingFace's free Inference API, so you don't need to manage your own GPU or pay for expensive API calls.

## How It Works

Three agents run in sequence for each clinical note:

| Step | Agent | What It Does | Output |
|------|-------|-------------|--------|
| 1 | **Extractor** | Reads a clinical note and picks out medical terms | List of conditions, medications, procedures, body sites |
| 2 | **Ontology Mapper** | Looks up each term in the SNOMED CT medical dictionary | Each term gets a standard code, label, and confidence score |
| 3 | **Reviewer** | Checks the results and writes a summary | Markdown report flagging any low-confidence or missing mappings |

After all notes are processed, a **validator** compares the extractions against expected answers and scores precision, recall, and F1.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/rtdatasci/clinical_data_ai.git
cd clinical_data_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your HuggingFace token

Create a `.env` file in the project root (this file is gitignored and never committed):

```
HF_TOKEN=hf_your_token_here
```

You can get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Run the pipeline

```bash
python run_pipeline.py
```

This processes all 5 clinical notes, saves results to `results/`, and prints an evaluation summary.

### 5. Run validation only

```bash
python evaluate/validator.py
```

Prints precision, recall, and F1 scores per category.

## Project Structure

```
clinical_data_ai/
├── data/sample_notes.json        # 5 synthetic clinical notes
├── agents/
│   ├── extractor.py              # Extracts medical terms from notes
│   ├── ontology_mapper.py        # Maps terms to SNOMED CT codes
│   └── reviewer.py               # Generates quality report
├── evaluate/
│   ├── ground_truth.json         # Expected answers for scoring
│   └── validator.py              # Computes precision/recall/F1
├── run_pipeline.py               # Runs the full pipeline
├── results/                      # Generated outputs (gitignored)
├── curation_agent.ipynb          # Reference: original genomics version
└── requirements.txt
```

## Sample Output

The extractor produces structured JSON like:

```json
{
  "patient_id": "P001",
  "conditions": ["type 2 diabetes", "hypertension"],
  "medications": ["metformin", "lisinopril"],
  "procedures": ["echocardiogram"],
  "body_sites": ["heart", "kidney"]
}
```

The mapper adds SNOMED codes:

```json
{
  "original_term": "type 2 diabetes",
  "snomed_id": "44054006",
  "snomed_label": "Type 2 diabetes mellitus",
  "confidence": 0.97
}
```

The reviewer flags anything with confidence below 0.90 for manual review.
