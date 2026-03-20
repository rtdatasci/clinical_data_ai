"""Main orchestrator: runs all 3 agents in sequence for each clinical note."""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from smolagents import InferenceClientModel

from agents.extractor import create_extractor_agent, extract_entities
from agents.ontology_mapper import create_mapper_agent, map_entities_to_snomed
from agents.reviewer import create_reviewer_agent, generate_report
from evaluate.validator import validate, print_results


def load_notes(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_text(text, path):
    with open(path, "w") as f:
        f.write(text)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load clinical notes
    notes_path = os.path.join(data_dir, "sample_notes.json")
    notes = load_notes(notes_path)
    print(f"Loaded {len(notes)} clinical notes.\n")

    # Create a shared model instance
    model = InferenceClientModel()

    # Create agents
    print("Initializing agents...")
    extractor = create_extractor_agent(model)
    mapper = create_mapper_agent(model)
    reviewer = create_reviewer_agent(model)
    print("Agents ready.\n")

    all_extractions = []
    all_mappings = {}
    all_reports = []

    for i, note in enumerate(notes):
        pid = note["patient_id"]
        print(f"{'='*60}")
        print(f"Processing {pid} ({i+1}/{len(notes)}) — {note['note_type']}")
        print(f"{'='*60}")

        # Step 1: Extract entities
        print(f"\n[1/3] Extracting clinical entities from {pid}...")
        try:
            extracted = extract_entities(extractor, note)
            print(f"  Extracted: {sum(len(extracted[k]) for k in ['conditions','medications','procedures','body_sites'])} entities")
        except Exception as e:
            print(f"  ERROR in extraction: {e}")
            extracted = {
                "patient_id": pid,
                "conditions": [], "medications": [],
                "procedures": [], "body_sites": [],
            }
        all_extractions.append(extracted)

        # Step 2: Map to SNOMED CT
        print(f"\n[2/3] Mapping entities to SNOMED CT for {pid}...")
        try:
            mappings = map_entities_to_snomed(mapper, extracted)
            mapped_count = sum(1 for m in mappings if m["snomed_id"] != "UNMAPPED")
            print(f"  Mapped: {mapped_count}/{len(mappings)} terms")
        except Exception as e:
            print(f"  ERROR in mapping: {e}")
            mappings = []
        all_mappings[pid] = mappings

        # Step 3: Generate review report
        print(f"\n[3/3] Generating review report for {pid}...")
        try:
            report = generate_report(reviewer, pid, mappings)
            print(f"  Report generated ({len(report)} chars)")
        except Exception as e:
            print(f"  ERROR in report generation: {e}")
            report = f"## Patient {pid}\n\nReport generation failed: {e}\n"
        all_reports.append(report)

        print()

    # Save all outputs
    print("Saving results...")
    save_json(all_extractions, os.path.join(results_dir, "extractions.json"))
    save_json(all_mappings, os.path.join(results_dir, "mappings.json"))

    # Combine all reports into one file
    combined_report = "\n\n---\n\n".join(all_reports)
    save_text(combined_report, os.path.join(results_dir, "reports.md"))

    print(f"  Extractions saved to results/extractions.json")
    print(f"  Mappings saved to results/mappings.json")
    print(f"  Reports saved to results/reports.md")

    # Step 4: Run validation
    print("\nRunning validation against ground truth...")
    gt_path = os.path.join(base_dir, "evaluate", "ground_truth.json")
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    extractions_dict = {e["patient_id"]: e for e in all_extractions}
    eval_results = validate(extractions_dict, ground_truth)
    print_results(eval_results)
    save_json(eval_results, os.path.join(results_dir, "evaluation.json"))
    print("Evaluation saved to results/evaluation.json")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
