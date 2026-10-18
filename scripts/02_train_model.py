"""
FairSkin AI — Model Training & Evaluation Pipeline.
Trains MobileNetV3-Small on the curated clinical dermatology dataset.
Optimizes multi-task loss (Disease classification + Observable features + Acne severity).
Saves the best checkpoint to `models/best_fairskin_model.pth`.
Evaluates on the strictly held-out test set and saves comprehensive metrics.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.models.backbone import FairSkinMobileNet, DISEASE_CLASSES, FEATURE_VOCABULARY
from src.data.dataset_loader import FairSkinDataset
from src.fairness.metrics import compute_comprehensive_metrics

MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "baseline_evaluation")
MANIFESTS_DIR = os.path.join(ROOT_DIR, "data", "manifests")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_epoch(model, dataloader, optimizer, criterion_disease, criterion_severity, criterion_features, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        d_targets = batch["disease_target"].to(device)
        s_targets = batch["severity_target"].to(device)
        f_targets = batch["feature_target"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # 1. Disease Loss
        l_disease = criterion_disease(outputs["disease_logits"], d_targets)

        # 2. Acne Severity Loss (only for acne samples)
        acne_mask = (d_targets == 0) & (s_targets >= 0)
        if torch.sum(acne_mask) > 0:
            l_severity = criterion_severity(outputs["severity_logits"][acne_mask], s_targets[acne_mask])
        else:
            l_severity = torch.tensor(0.0, device=device)

        # 3. Observable Features Loss
        l_features = criterion_features(outputs["feature_logits"], f_targets)

        # Total combined loss
        loss = l_disease + 0.3 * l_severity + 0.5 * l_features
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        preds = torch.argmax(outputs["disease_logits"], dim=1)
        correct += (preds == d_targets).sum().item()
        total += len(images)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_subgroups = []

    for batch in dataloader:
        images = batch["image"].to(device)
        d_targets = batch["disease_target"].to(device)
        fst = batch["fitzpatrick_scale"]

        outputs = model(images)
        probs = torch.softmax(outputs["disease_logits"], dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(d_targets.cpu().numpy())
        all_probs.extend(probs)
        all_subgroups.extend([int(x) for x in fst])

    return (

        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        all_subgroups
    )


def main():
    print("=" * 70)
    print("FAIRSKIN AI — MODEL TRAINING & SCIENTIFIC EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")

    train_csv = os.path.join(MANIFESTS_DIR, "train_split.csv")
    val_csv = os.path.join(MANIFESTS_DIR, "val_split.csv")
    test_csv = os.path.join(MANIFESTS_DIR, "test_split.csv")

    train_dataset = FairSkinDataset(train_csv, is_train=True)
    val_dataset = FairSkinDataset(val_csv, is_train=False)
    test_dataset = FairSkinDataset(test_csv, is_train=False)

    # Balanced sampler to ensure equal probability of selecting each disease class per batch
    class_counts = train_dataset.df["disease_label"].value_counts().to_dict()
    weights = []
    for c in DISEASE_CLASSES:
        cnt = class_counts.get(c, 1)
        weights.append(1.0 / max(cnt, 1))
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(DISEASE_CLASSES)

    sample_weights = [weights_tensor[DISEASE_CLASSES.index(l)].item() for l in train_dataset.df["disease_label"]]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Dataset Loaded: {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Held-Out Test", flush=True)

    # Initialize model with warm-start fine-tuning if checkpoint exists
    model = FairSkinMobileNet(pretrained=True).to(device)
    best_model_path = os.path.join(MODELS_DIR, "best_fairskin_model.pth")
    if os.path.exists(best_model_path):
        try:
            print(f"Loading pre-trained baseline weights from {best_model_path} for fine-tuning...", flush=True)
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("Successfully loaded pre-trained baseline weights.", flush=True)
        except Exception as e:
            print(f"Notice: initialized fresh backbone ({e})", flush=True)

    criterion_disease = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.08)
    criterion_severity = nn.CrossEntropyLoss()
    criterion_features = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    epochs = 14
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0

    print(f"\nBeginning training for {epochs} epochs with balanced batch sampling...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(
            model, train_loader, optimizer,
            criterion_disease, criterion_severity, criterion_features, device
        )
        scheduler.step()

        val_targets, val_preds, val_probs, val_subgroups = evaluate(model, val_loader, device)
        val_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs, val_subgroups)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}", flush=True)

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "classes": DISEASE_CLASSES,
                "features": FEATURE_VOCABULARY
            }, best_model_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best model checkpoint saved to: {best_model_path}", flush=True)

    # Final Evaluation on the untouched held-out test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET")
    print("=" * 70)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_targets, test_preds, test_probs, test_subgroups = evaluate(model, test_loader, device)
    test_metrics = compute_comprehensive_metrics(test_targets, test_preds, test_probs, test_subgroups)

    print(f"Test Accuracy:                 {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Balanced Accuracy:        {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Test Macro-F1:                 {test_metrics['macro_f1']:.4f}")
    print(f"Worst-Group Sensitivity (WGS): {test_metrics['worst_group_sensitivity']*100:.2f}%")
    print(f"Mean Equal Opportunity Diff:   {test_metrics['mean_equal_opportunity_diff']:.4f}")
    print(f"Expected Calibration Error:    {test_metrics['expected_calibration_error']:.4f}")

    print("\nPer-Class Performance Breakdown:")
    for c_name in DISEASE_CLASSES:
        c_stats = test_metrics["per_class_report"].get(c_name, {})
        prec = c_stats.get("precision", 0) * 100
        rec = c_stats.get("recall", 0) * 100
        f1 = c_stats.get("f1-score", 0)
        supp = c_stats.get("support", 0)
        print(f"  {c_name:<16}: Precision: {prec:5.1f}% | Recall: {rec:5.1f}% | F1: {f1:.3f} (n={supp})")

    # Save metrics JSON
    metrics_out = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nComplete evaluation metrics report saved to: {metrics_out}")


if __name__ == "__main__":
    main()
