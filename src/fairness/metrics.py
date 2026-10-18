"""
Fairness and Clinical Evaluation Metrics.
Calculates:
  - Macro-F1, Balanced Accuracy, Micro-Accuracy
  - Per-disease Precision, Recall, Specificity, F1
  - Per-Skin-Tone Subgroup Disparities
  - Worst-Group Sensitivity (WGS)
  - Equal Opportunity Difference (Delta EOD)
  - Expected Calibration Error (ECE)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score

from src.models.backbone import DISEASE_CLASSES


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subgroups: List[int],
    class_names: List[str] = DISEASE_CLASSES
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )

    # Subgroup (Skin Tone) Analysis
    subgroups_arr = np.array([int(s) for s in subgroups], dtype=int)
    unique_subgroups = sorted(list(set(subgroups_arr.tolist())))
    subgroup_metrics = {}
    recalls_by_subgroup = {s: [] for s in unique_subgroups}

    for s in unique_subgroups:
        s_mask = (subgroups_arr == s)
        if int(np.sum(s_mask)) == 0:
            continue

        s_true = y_true[s_mask]
        s_pred = y_pred[s_mask]
        s_acc = accuracy_score(s_true, s_pred)
        s_f1 = f1_score(s_true, s_pred, average="macro", zero_division=0)

        subgroup_metrics[f"FST_{s}"] = {
            "sample_count": int(np.sum(s_mask)),
            "accuracy": round(float(s_acc), 4),
            "macro_f1": round(float(s_f1), 4)
        }

    # Per-disease, per-skin-type recall matrix
    disease_fst_matrix = {}
    min_recall = 1.0
    worst_group = ("None", -1)
    eod_per_disease = {}

    for c_idx, c_name in enumerate(class_names):
        c_recalls = {}
        for s in unique_subgroups:
            cell_mask = (y_true == c_idx) & (subgroups_arr == s)
            cell_count = int(np.sum(cell_mask))

            if cell_count > 0:
                cell_recall = np.sum((y_pred == c_idx) & cell_mask) / float(cell_count)
                c_recalls[f"FST_{s}"] = round(float(cell_recall), 4)
                if cell_recall < min_recall:
                    min_recall = cell_recall
                    worst_group = (c_name, s)
            else:
                c_recalls[f"FST_{s}"] = None

        valid_recalls = [v for v in c_recalls.values() if v is not None]
        if len(valid_recalls) >= 2:
            eod_per_disease[c_name] = round(float(max(valid_recalls) - min(valid_recalls)), 4)
        else:
            eod_per_disease[c_name] = 0.0

        disease_fst_matrix[c_name] = c_recalls

    # Expected Calibration Error (ECE)
    max_probs = np.max(y_prob, axis=1)
    correctness = (y_pred == y_true).astype(float)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(correctness[bin_mask])
            bin_conf = np.mean(max_probs[bin_mask])
            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)

    mean_delta_eod = float(np.mean(list(eod_per_disease.values()))) if eod_per_disease else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "expected_calibration_error": round(float(ece), 4),
        "worst_group_sensitivity": round(float(min_recall), 4),
        "worst_group": {"disease": worst_group[0], "fst": worst_group[1]},
        "mean_equal_opportunity_diff": round(mean_delta_eod, 4),
        "eod_by_disease": eod_per_disease,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "subgroup_metrics": subgroup_metrics,
        "disease_fst_matrix": disease_fst_matrix
    }
