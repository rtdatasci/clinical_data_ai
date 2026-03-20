"""Agent 1: Extract clinical entities from clinical notes."""

import json
from smolagents import CodeAgent, InferenceClientModel


EXTRACTION_PROMPT = """You are a clinical NLP specialist. Extract structured clinical entities from the following clinical note.

Return ONLY a valid JSON object with these keys:
- "conditions": list of medical conditions, diseases, or diagnoses mentioned
- "medications": list of medication names (without dosages)
- "procedures": list of medical procedures, tests, or interventions
- "body_sites": list of anatomical locations or body parts mentioned

Be thorough but only include entities clearly mentioned in the text.
Normalize terms to lowercase.

Clinical note:
{note_text}

Return ONLY the JSON object, no other text."""


def create_extractor_agent(model=None):
    """Create a CodeAgent for clinical entity extraction."""
    if model is None:
        model = InferenceClientModel()
    agent = CodeAgent(
        tools=[],
        model=model,
        name="clinical_extractor",
        description="Extracts clinical entities (conditions, medications, procedures, body sites) from clinical notes.",
    )
    return agent


def extract_entities(agent, note):
    """Run the extractor agent on a single clinical note.

    Args:
        agent: A smolagents CodeAgent instance.
        note: dict with keys patient_id, note_type, text.

    Returns:
        dict with patient_id and extracted entity lists.
    """
    prompt = EXTRACTION_PROMPT.format(note_text=note["text"])
    result = agent.run(prompt)

    # Parse the agent's response into structured JSON
    extracted = _parse_extraction(result)
    extracted["patient_id"] = note["patient_id"]
    return extracted


def _parse_extraction(result):
    """Parse agent output into a structured dict."""
    # If already a dict, use it directly
    if isinstance(result, dict):
        return _normalize(result)

    text = str(result)

    # Try to find JSON in the response
    # Look for JSON block between ```json and ```
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try to find a JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        parsed = json.loads(text)
        return _normalize(parsed)
    except json.JSONDecodeError:
        return {
            "conditions": [],
            "medications": [],
            "procedures": [],
            "body_sites": [],
        }


def _normalize(data):
    """Ensure all expected keys exist and values are lowercase lists."""
    result = {}
    for key in ["conditions", "medications", "procedures", "body_sites"]:
        values = data.get(key, [])
        if isinstance(values, list):
            result[key] = [str(v).lower().strip() for v in values]
        else:
            result[key] = []
    return result
