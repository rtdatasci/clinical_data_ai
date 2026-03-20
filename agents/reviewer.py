"""Agent 3: Generate a human-readable review report from SNOMED mappings."""

from smolagents import CodeAgent, InferenceClientModel


REVIEW_PROMPT = """You are a clinical data quality reviewer. Generate a concise Markdown report for the following patient's clinical data extraction and SNOMED CT mapping results.

Patient ID: {patient_id}

Extracted entities and their SNOMED mappings:
{mappings_text}

Generate a Markdown report with these sections:

## Patient {patient_id} — Clinical Data Extraction Report

### Summary Statistics
- Total terms extracted
- Total terms successfully mapped (snomed_id is not "UNMAPPED")
- Number of low-confidence mappings (confidence < 0.90)

### Mapping Table
A markdown table with columns: Original Term | Category | SNOMED ID | SNOMED Label | Confidence

### Flagged Items
List any terms with confidence < 0.90 or that are UNMAPPED, with a brief note on why they may need manual review.

### Overall Assessment
One sentence on the quality of the extraction and mapping.

Return ONLY the Markdown report."""


def create_reviewer_agent(model=None):
    """Create a CodeAgent for generating review reports."""
    if model is None:
        model = InferenceClientModel()
    agent = CodeAgent(
        tools=[],
        model=model,
        name="clinical_reviewer",
        description="Generates human-readable review reports for clinical data extractions.",
    )
    return agent


def generate_report(agent, patient_id, mappings):
    """Generate a Markdown review report.

    Args:
        agent: A smolagents CodeAgent.
        patient_id: str patient identifier.
        mappings: list of SNOMED mapping dicts.

    Returns:
        str: Markdown report.
    """
    if not mappings:
        return f"## Patient {patient_id}\n\nNo entities were extracted for this patient.\n"

    mappings_text = ""
    for m in mappings:
        mappings_text += (
            f"- Term: {m['original_term']}, Category: {m['category']}, "
            f"SNOMED ID: {m['snomed_id']}, Label: {m['snomed_label']}, "
            f"Confidence: {m['confidence']}\n"
        )

    prompt = REVIEW_PROMPT.format(
        patient_id=patient_id,
        mappings_text=mappings_text,
    )

    result = agent.run(prompt)
    report = str(result)

    # Clean up code fences if the agent wrapped the markdown
    if report.startswith("```markdown"):
        report = report[len("```markdown"):].strip()
    if report.startswith("```"):
        report = report[3:].strip()
    if report.endswith("```"):
        report = report[:-3].strip()

    return report
