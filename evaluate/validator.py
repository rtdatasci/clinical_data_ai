"""Validate agent extractions against ground truth and compute metrics."""

import json
import os
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_term(term):
    """Normalize a term for comparison."""
    return term.lower().strip()


def compute_metrics(predicted, expected):
    """Compute precision, recall, F1 for two sets of terms.

    Args:
        predicted: list of predicted terms.
        expected: list of expected (ground truth) terms.

    Returns:
        dict with precision, recall, f1, tp, fp, fn.
    """
    pred_set = {normalize_term(t) for t in predicted}
    exp_set = {normalize_term(t) for t in expected}

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": sorted(pred_set),
        "expected": sorted(exp_set),
    }


def validate(extractions, ground_truth):
    """Compare extractions against ground truth for all patients.

    Args:
        extractions: dict mapping patient_id -> extracted entities dict.
        ground_truth: dict mapping patient_id -> expected entities dict.

    Returns:
        dict with per-patient and aggregate metrics.
    """
    categories = ["conditions", "medications", "procedures", "body_sites"]
    results = {"per_patient": {}, "aggregate": {}}

    # Aggregate counters
    agg = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in categories}

    for patient_id, expected in ground_truth.items():
        predicted = extractions.get(patient_id, {})
        patient_results = {}

        for cat in categories:
            pred_list = predicted.get(cat, [])
            exp_list = expected.get(cat, [])
            metrics = compute_metrics(pred_list, exp_list)
            patient_results[cat] = metrics

            agg[cat]["tp"] += metrics["tp"]
            agg[cat]["fp"] += metrics["fp"]
            agg[cat]["fn"] += metrics["fn"]

        results["per_patient"][patient_id] = patient_results

    # Compute aggregate metrics
    for cat in categories:
        tp = agg[cat]["tp"]
        fp = agg[cat]["fp"]
        fn = agg[cat]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results["aggregate"][cat] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


def print_results(results):
    """Print a formatted evaluation table."""
    categories = ["conditions", "medications", "procedures", "body_sites"]

    print("\n" + "=" * 70)
    print("CLINICAL DATA EXTRACTION — EVALUATION RESULTS")
    print("=" * 70)

    # Per-patient results
    for patient_id in sorted(results["per_patient"].keys()):
        print(f"\n--- {patient_id} ---")
        print(f"  {'Category':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'-'*65}")
        for cat in categories:
            m = results["per_patient"][patient_id].get(cat, {})
            print(
                f"  {cat:<15} {m.get('precision', 0):.3f}      "
                f"{m.get('recall', 0):.3f}      {m.get('f1', 0):.3f}      "
                f"{m.get('tp', 0):>3}   {m.get('fp', 0):>3}   {m.get('fn', 0):>3}"
            )

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Category':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*65}")
    for cat in categories:
        m = results["aggregate"].get(cat, {})
        print(
            f"  {cat:<15} {m.get('precision', 0):.3f}      "
            f"{m.get('recall', 0):.3f}      {m.get('f1', 0):.3f}      "
            f"{m.get('tp', 0):>3}   {m.get('fp', 0):>3}   {m.get('fn', 0):>3}"
        )
    print()


def main():
    """Run validation standalone using saved extraction results."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_path = os.path.join(base_dir, "evaluate", "ground_truth.json")
    results_dir = os.path.join(base_dir, "results")
    extractions_path = os.path.join(results_dir, "extractions.json")

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}")
        sys.exit(1)

    if not os.path.exists(extractions_path):
        print(f"Error: Extractions file not found at {extractions_path}")
        print("Run `python run_pipeline.py` first to generate extractions.")
        sys.exit(1)

    ground_truth = load_json(gt_path)
    extractions_list = load_json(extractions_path)

    # Convert list format to dict keyed by patient_id
    extractions = {}
    for item in extractions_list:
        pid = item.get("patient_id")
        if pid:
            extractions[pid] = item

    results = validate(extractions, ground_truth)
    print_results(results)

    # Save results
    output_path = os.path.join(results_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
