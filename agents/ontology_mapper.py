"""Agent 2: Map extracted clinical entities to SNOMED CT codes."""

import json
import requests
from smolagents import CodeAgent, InferenceClientModel, tool


SNOMED_BROWSER_API = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions"


@tool
def search_snomed(term: str) -> str:
    """Search the SNOMED CT terminology for a clinical term and return matching concepts.

    Args:
        term: The clinical term to search for (e.g., 'type 2 diabetes', 'metformin').

    Returns:
        A JSON string with the top SNOMED CT match including concept ID and label,
        or an error message if no match is found.
    """
    try:
        params = {
            "term": term,
            "limit": 5,
            "active": True,
            "semanticTag": "",
            "language": "en",
        }
        headers = {"Accept": "application/json"}
        resp = requests.get(SNOMED_BROWSER_API, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            return json.dumps({"error": f"No SNOMED CT match found for '{term}'"})

        # Return top matches
        matches = []
        for item in items[:3]:
            concept = item.get("concept", {})
            matches.append({
                "term": item.get("term", ""),
                "concept_id": concept.get("conceptId", ""),
                "active": concept.get("active", False),
                "semantic_tag": item.get("semanticTag", ""),
            })
        return json.dumps({"query": term, "matches": matches})

    except requests.RequestException as e:
        return json.dumps({"error": f"SNOMED API request failed: {str(e)}"})


def create_mapper_agent(model=None):
    """Create a CodeAgent for SNOMED CT ontology mapping."""
    if model is None:
        model = InferenceClientModel()
    agent = CodeAgent(
        tools=[search_snomed],
        model=model,
        name="ontology_mapper",
        description="Maps clinical terms to SNOMED CT codes using the SNOMED browser API.",
    )
    return agent


MAPPING_PROMPT = """You are a clinical terminology specialist. Map each clinical term to its SNOMED CT code.

For each term in the list below, use the search_snomed tool to find the best SNOMED CT match.

Terms to map:
{terms}

For each term, call search_snomed with the term, then pick the best match from the results.

Return a JSON list where each item has:
- "original_term": the input term
- "snomed_id": the SNOMED CT concept ID (string)
- "snomed_label": the official SNOMED CT label
- "confidence": a float 0.0-1.0 indicating match quality (1.0 = exact match, lower for partial)
- "category": one of "condition", "medication", "procedure", "body_site"

If no match is found, set snomed_id to "UNMAPPED", snomed_label to "No match found", and confidence to 0.0.

Return ONLY the JSON list."""


def map_entities_to_snomed(agent, extracted):
    """Map all extracted entities to SNOMED CT.

    Args:
        agent: A smolagents CodeAgent with search_snomed tool.
        extracted: dict with conditions, medications, procedures, body_sites lists.

    Returns:
        list of dicts with SNOMED mappings.
    """
    all_terms = []
    for category, key in [
        ("condition", "conditions"),
        ("medication", "medications"),
        ("procedure", "procedures"),
        ("body_site", "body_sites"),
    ]:
        for term in extracted.get(key, []):
            all_terms.append({"term": term, "category": category})

    if not all_terms:
        return []

    terms_text = "\n".join(
        f"- {t['term']} (category: {t['category']})" for t in all_terms
    )
    prompt = MAPPING_PROMPT.format(terms=terms_text)
    result = agent.run(prompt)

    mappings = _parse_mappings(result, all_terms)
    return mappings


def _parse_mappings(result, all_terms):
    """Parse agent output into a list of SNOMED mappings."""
    if isinstance(result, list):
        return _normalize_mappings(result, all_terms)

    text = str(result)

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_mappings(parsed, all_terms)
    except json.JSONDecodeError:
        pass

    # Fallback: create unmapped entries for all terms
    return [
        {
            "original_term": t["term"],
            "snomed_id": "UNMAPPED",
            "snomed_label": "No match found",
            "confidence": 0.0,
            "category": t["category"],
        }
        for t in all_terms
    ]


def _normalize_mappings(mappings, all_terms):
    """Ensure all mappings have required fields."""
    normalized = []
    for m in mappings:
        if not isinstance(m, dict):
            continue
        normalized.append({
            "original_term": m.get("original_term", ""),
            "snomed_id": str(m.get("snomed_id", "UNMAPPED")),
            "snomed_label": m.get("snomed_label", "No match found"),
            "confidence": float(m.get("confidence", 0.0)),
            "category": m.get("category", "unknown"),
        })
    return normalized
