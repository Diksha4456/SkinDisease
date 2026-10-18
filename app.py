"""
FairSkin AI — Fully Offline Skin Disease Screening Application.
Offline-First Mobile & Web Interface.
Zero cloud dependencies — 100% local inference.
Supported Targets: Acne (Low/High), Melasma, Vitiligo, Psoriasis, Ringworm, and Non-Target/OOD.
"""

import os
import sys
import io
import base64
import json
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Project Root setup
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.models.backbone import FairSkinMobileNet, DISEASE_CLASSES, FEATURE_VOCABULARY
from src.data.skin_tone_estimator import estimate_ita
from src.data.quality_guard import ImageQualityGuard
from src.xai.gradcam import GradCAMExplainer, build_evidence_explanation
from src.data.dataset_loader import get_transforms

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global Model & Components Initialization
DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_fairskin_model.pth")

print(f"Loading FairSkin AI Model from: {MODEL_PATH}...")
model = FairSkinMobileNet(pretrained=False).to(DEVICE)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded model (Validation Macro-F1: {checkpoint.get('val_macro_f1', 0):.4f})")
else:
    print("Warning: Model checkpoint not found! Using initialized backbone.")

model.eval()
explainer = GradCAMExplainer(model)
quality_guard = ImageQualityGuard()
infer_transforms = get_transforms(is_train=False, image_size=224)
from torchvision import transforms as TVTransforms
full_frame_infer_tf = TVTransforms.Compose([
    TVTransforms.Resize((224, 224)),
    TVTransforms.ToTensor(),
    TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FairSkin AI — Offline Skin Disease Screening</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #0284c7;
      --primary-hover: #0369a1;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-family);
      line-height: 1.5;
      padding: 16px;
      min-height: 100vh;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    header {
      text-align: center;
      margin-bottom: 24px;
      padding: 16px 0;
      border-bottom: 1px solid var(--border);
    }
    .badge-offline {
      display: inline-block;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    h1 {
      font-size: 1.75rem;
      font-weight: 700;
      color: #fff;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-top: 4px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }
    @media (min-width: 768px) {
      .grid { grid-template-columns: 1fr 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(2, 132, 199, 0.05);
    }
    .dropzone-icon {
      font-size: 2.2rem;
      color: var(--primary);
      margin-bottom: 8px;
    }
    .btn {
      display: inline-block;
      width: 100%;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 6px;
      padding: 12px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
      margin-top: 12px;
      text-align: center;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { background: var(--border); cursor: not-allowed; }
    #preview-container {
      margin-top: 16px;
      text-align: center;
      display: none;
    }
    #preview-img {
      max-width: 100%;
      max-height: 240px;
      border-radius: 8px;
      border: 1px solid var(--border);
      object-fit: cover;
    }
    .status-alert {
      padding: 12px;
      border-radius: 6px;
      margin-bottom: 16px;
      font-size: 0.85rem;
      display: none;
    }
    .status-alert.warning {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.4);
      color: #fbbf24;
    }
    .status-alert.danger {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }
    .report-header {
      background: rgba(2, 132, 199, 0.1);
      border: 1px solid rgba(2, 132, 199, 0.3);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
    }
    .disease-title {
      font-size: 1.4rem;
      font-weight: 700;
      color: #38bdf8;
    }
    .conf-bar-bg {
      background: #334155;
      height: 8px;
      border-radius: 4px;
      margin-top: 6px;
      overflow: hidden;
    }
    .conf-bar-fill {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.5s ease;
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 0.85rem;
    }
    .meta-label { color: var(--text-muted); }
    .meta-val { font-weight: 600; }
    .features-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .feature-tag {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #cbd5e1;
    }
    .xai-views {
      display: flex;
      gap: 12px;
      margin-top: 14px;
    }
    .xai-thumb {
      flex: 1;
      text-align: center;
    }
    .xai-thumb img {
      width: 100%;
      height: 130px;
      border-radius: 6px;
      border: 1px solid var(--border);
      object-fit: cover;
    }
    .xai-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .disclaimer {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 16px;
      padding: 10px;
      border-left: 3px solid var(--warning);
      background: rgba(255, 255, 255, 0.03);
    }
    .loader {
      display: none;
      text-align: center;
      padding: 24px;
    }
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      border-left-color: var(--primary);
      animation: spin 1s linear infinite;
      display: inline-block;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge-offline">&#10003; 100% Offline Inference &mdash; Airplane Mode Verified</span>
      <h1>FairSkin AI Screening</h1>
      <p class="subtitle">Fair &amp; Evidence-Grounded Screening across Fitzpatrick Tones I&ndash;VI</p>
    </header>

    <div class="grid">
      <!-- Input Card -->
      <div class="card">
        <div class="card-title">&#128247; Clinical Image Capture</div>
        <div class="dropzone" id="dropzone">
          <div class="dropzone-icon">&#8682;</div>
          <div style="font-weight: 600;">Drag &amp; Drop Skin Photo</div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">or click to select from local storage</div>
          <input type="file" id="file-input" accept="image/*" style="display: none;">
        </div>

        <div id="preview-container">
          <img id="preview-img" src="" alt="Selected skin lesion preview">
        </div>

        <button class="btn" id="analyze-btn" disabled>Run Offline Screening</button>

        <div class="disclaimer">
          <strong>Clinical Safety Protocol:</strong> This on-device tool conducts automated preliminary screening for 5 conditions (Acne, Melasma, Psoriasis, Ringworm, Vitiligo). It rejects low-quality or ambiguous inputs and never forces an ungrounded diagnosis.
        </div>
      </div>

      <!-- Results Card -->
      <div class="card" id="results-card">
        <div class="card-title">&#128202; Diagnostic Screening Report</div>

        <div class="loader" id="loader">
          <div class="spinner"></div>
          <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">Analyzing Image Quality, Skin Tone &amp; Lesion Morphology...</div>
        </div>

        <div id="status-box" class="status-alert"></div>

        <div id="results-content" style="display: none;">
          <div class="report-header">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="disease-title" id="res-disease">Acne Vulgaris</span>
              <span id="res-severity" style="font-size: 0.85rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: rgba(56, 189, 248, 0.2); color: #38bdf8;">Low Severity</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); margin-top: 8px;">
              <span>Softmax Confidence</span>
              <span id="res-conf-txt">92.4%</span>
            </div>
            <div class="conf-bar-bg">
              <div class="conf-bar-fill" id="res-conf-bar"></div>
            </div>
          </div>

          <div class="meta-row">
            <span class="meta-label">Objective Skin Tone (ITA):</span>
            <span class="meta-val" id="res-tone">Tan (FST IV) / 22.4&deg;</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Image Quality Check:</span>
            <span class="meta-val" id="res-qc" style="color: #34d399;">Passed (Sharp, Well-Lit)</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Prediction Uncertainty (Entropy):</span>
            <span class="meta-val" id="res-entropy">0.142 (Low)</span>
          </div>

          <div style="margin-top: 14px;">
            <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Identified Visual Features:</div>
            <div class="features-list" id="res-features"></div>
          </div>

          <div style="margin-top: 14px;">
            <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Explainable AI (Grad-CAM Evidence):</div>
            <div class="xai-views">
              <div class="xai-thumb">
                <img id="res-xai-orig" src="" alt="Original lesion">
                <div class="xai-label">Original Input</div>
              </div>
              <div class="xai-thumb">
                <img id="res-xai-overlay" src="" alt="Grad-CAM activation overlay">
                <div class="xai-label">Grad-CAM Salience</div>
              </div>
            </div>
          </div>

          <div style="margin-top: 14px; font-size: 0.8rem; line-height: 1.4; color: #cbd5e1; background: rgba(255,255,255,0.03); padding: 10px; border-radius: 6px;">
            <strong>Clinical Rationale:</strong>
            <p id="res-rationale" style="margin-top: 4px;"></p>
          </div>
        </div>

        <div id="empty-state" style="text-align: center; color: var(--text-muted); padding: 48px 16px;">
          <div style="font-size: 2rem; margin-bottom: 8px;">&#128065;</div>
          <div>No image analyzed yet.</div>
          <div style="font-size: 0.8rem; margin-top: 4px;">Upload a skin photograph on the left to view the clinical screening report.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('results-content');
    const emptyState = document.getElementById('empty-state');
    const statusBox = document.getElementById('status-box');

    let currentFile = null;

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPEG, PNG, WEBP).');
        return;
      }
      currentFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = 'Analyze ' + file.name.substring(0, 20);
      };
      reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) return;

      analyzeBtn.disabled = true;
      loader.style.display = 'block';
      resultsContent.style.display = 'none';
      emptyState.style.display = 'none';
      statusBox.style.display = 'none';

      const formData = new FormData();
      formData.append('image', currentFile);

      try {
        const resp = await fetch('/api/screen', {
          method: 'POST',
          body: formData
        });

        const data = await resp.json();
        loader.style.display = 'none';
        analyzeBtn.disabled = false;

        if (!data.success) {
          statusBox.className = 'status-alert danger';
          statusBox.innerText = data.error || 'Screening failed.';
          statusBox.style.display = 'block';
          return;
        }

        // Quality warning if applicable
        if (data.quality_warning) {
          statusBox.className = 'status-alert warning';
          statusBox.innerText = 'Quality Warning: ' + data.quality_warning;
          statusBox.style.display = 'block';
        }

        // Populate report UI
        document.getElementById('res-disease').innerText = data.predicted_disease;
        const sevElem = document.getElementById('res-severity');
        if (data.predicted_disease.toLowerCase() === 'acne' && data.acne_severity !== 'N/A') {
          sevElem.innerText = data.acne_severity + ' Severity';
          sevElem.style.display = 'inline-block';
        } else {
          sevElem.style.display = 'none';
        }

        document.getElementById('res-conf-txt').innerText = data.confidence_pct + '%';
        document.getElementById('res-conf-bar').style.width = data.confidence_pct + '%';

        document.getElementById('res-tone').innerText = `${data.ita_tone_band} (FST ${data.fitzpatrick_scale}) / ${data.ita_degrees}\u00B0`;
        document.getElementById('res-entropy').innerText = `${data.entropy} (${data.uncertainty_level})`;
        document.getElementById('res-rationale').innerText = data.clinical_rationale;

        // Observable Features
        const featContainer = document.getElementById('res-features');
        featContainer.innerHTML = '';
        data.observable_features.forEach(f => {
          const tag = document.createElement('span');
          tag.className = 'feature-tag';
          tag.innerText = f;
          featContainer.appendChild(tag);
        });

        // XAI Visuals
        document.getElementById('res-xai-orig').src = 'data:image/jpeg;base64,' + data.original_b64;
        document.getElementById('res-xai-overlay').src = 'data:image/jpeg;base64,' + data.gradcam_b64;

        resultsContent.style.display = 'block';

      } catch (err) {
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
        statusBox.className = 'status-alert danger';
        statusBox.innerText = 'Network/Inference error: ' + err.message;
        statusBox.style.display = 'block';
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "offline_ready": True,
        "device": str(DEVICE),
        "target_diseases": DISEASE_CLASSES[:5],
        "acne_severity_supported": True
    })


def _run_clinical_screening(pil_img: Image.Image, raw_bytes: bytes):
    # 1. Image Quality Pre-Flight
    qc_result = quality_guard.check_quality(pil_img)
    quality_warning = None
    if not qc_result["passed"]:
        quality_warning = "; ".join(qc_result["flags"])

    # 2. Objective Skin Tone (ITA & FST)
    ita_deg, tone_band, fst = estimate_ita(pil_img)

    # 3. Model Inference (Un-cropped Macro + Multi-Scale Lesion Ensemble)
    w, h = pil_img.size
    t_full = full_frame_infer_tf(pil_img).unsqueeze(0).to(DEVICE)
    sub_crops = [
        pil_img,
        pil_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92))),
        pil_img.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.70))),
        pil_img.crop((int(w * 0.15), int(h * 0.30), int(w * 0.85), h))
    ]
    crop_tensors = torch.cat([t_full, torch.stack([infer_transforms(c) for c in sub_crops]).to(DEVICE)], dim=0)
    with torch.no_grad():
        multi_outs = model(crop_tensors)
        avg_disease_probs = multi_outs["disease_probs"].mean(dim=0).cpu().numpy()
        avg_feat_probs = multi_outs["feature_probs"].mean(dim=0).cpu().numpy()

    # Clinical Knowledge Prior: Differential diagnosis grounding based on pathognomonic visual signs
    # DISEASE_CLASSES: ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]
    feat_evidence = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
    # 0: Acne: papules/pustules (idx 0), comedones (idx 1), nodules/scars (idx 2)
    acne_score = float(max(avg_feat_probs[0], avg_feat_probs[1]) * 0.70 + avg_feat_probs[2] * 0.30)
    # 1: Melasma: reticulated hyperpigmentation (idx 3), irregular pigment border (idx 4)
    mel_score = float(max(avg_feat_probs[3], avg_feat_probs[4]))
    # 2: Psoriasis: erythematous scaly plaque (idx 7), micaceous silvery scale (idx 8)
    pso_score = float(max(avg_feat_probs[7], avg_feat_probs[8]))
    # 3: Ringworm: annular erythematous ring (idx 9), active advancing border (idx 10), central clearing (idx 11)
    rw_score = float(max(avg_feat_probs[9], avg_feat_probs[10]) * 0.70 + avg_feat_probs[11] * 0.30)
    # 4: Vitiligo: chalky white depigmentation (idx 5), well-demarcated border (idx 6)
    vit_score = float(max(avg_feat_probs[5], avg_feat_probs[6]))

    # Differential Diagnosis Constraints:
    if acne_score > 0.30:
        rw_score *= 0.20
        pso_score *= 0.35
    if mel_score > 0.35:
        rw_score *= 0.20
        vit_score *= 0.10
    if vit_score > 0.40:
        mel_score *= 0.10
    if pso_score > 0.40:
        acne_score *= 0.25

    max_target_feat = max(acne_score, mel_score, pso_score, rw_score, vit_score)
    ood_feat = max(0.05, 1.0 - max_target_feat * 2.5) if max_target_feat < 0.25 else 0.05

    feat_evidence[0] = acne_score
    feat_evidence[1] = mel_score
    feat_evidence[2] = pso_score
    feat_evidence[3] = rw_score
    feat_evidence[4] = vit_score
    feat_evidence[5] = ood_feat

    fused_probs = 0.70 * avg_disease_probs + 0.30 * feat_evidence
    probs = fused_probs / fused_probs.sum()
    pred_idx = int(np.argmax(probs))
    pred_disease = DISEASE_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Prediction Entropy
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
    uncertainty_level = "High" if entropy > 1.2 else ("Moderate" if entropy > 0.6 else "Low")

    # 4. Gated Acne Severity
    acne_severity = "N/A"
    if pred_disease == "acne":
        sev_probs = torch.softmax(multi_outs["severity_logits"].mean(dim=0), dim=-1).cpu().numpy()
        acne_severity = "High" if sev_probs[1] > sev_probs[0] else "Low"

    # 5. Observable Features Probabilities
    feat_dict = {FEATURE_VOCABULARY[i]: float(avg_feat_probs[i]) for i in range(len(FEATURE_VOCABULARY))}

    # 6. Grad-CAM++ Visualization with exact full-frame coordinate mapping
    from torchvision import transforms as TVTransforms
    full_frame_tf = TVTransforms.Compose([
        TVTransforms.Resize((224, 224)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gradcam_tensor = full_frame_tf(pil_img).unsqueeze(0).to(DEVICE)
    heatmap = explainer.generate_heatmap(gradcam_tensor, pred_idx)
    overlay_img = explainer.overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45)

    # 7. Evidence-Grounded Explanation
    explanation = build_evidence_explanation(
        disease=pred_disease,
        confidence=confidence,
        entropy=entropy,
        feature_probs=feat_dict,
        severity=acne_severity
    )

    # 5 Monitored Conditions Scope Enforcement
    TARGET_5 = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
    raw_ood_prob = float(avg_disease_probs[5])

    # If the presentation does not match any of the 5 monitored conditions,
    # or if the model's confidence in a target condition is insufficient (< 50%),
    # or pathognomonic visual feature evidence is absent (< 0.25):
    # The system strictly returns "Not Detected / Not Recognized" and does NOT guess from the 5 diseases.
    is_outside_target = (
        pred_disease == "non_target_ood"
        or raw_ood_prob >= 0.28
        or (pred_disease in TARGET_5 and confidence < 0.50)
        or (pred_disease in TARGET_5 and max_target_feat < 0.25)
    )

    if is_outside_target:
        display_disease = "Not Detected / Not Recognized"
        acne_severity = "N/A"
        explanation["observable_features"] = ["No Monitored Lesions Confirmed"]
        explanation["clinical_rationale"] = (
            "The presented skin lesion does NOT match any of the 5 target monitored conditions "
            "(Acne Vulgaris, Melasma, Plaque Psoriasis, Ringworm, or Vitiligo). "
            "No pathognomonic visual markers for target conditions were detected. "
            "A formal clinical evaluation by a board-certified dermatologist is recommended for any non-target rash or lesion."
        )
        explanation["referral_guidance"] = (
            "Condition is outside the 5 monitored screening categories. Consultation with a dermatologist recommended."
        )
    else:
        display_disease = pred_disease.title()

    # 8. Data Warehouse (DWH) Streaming Ingestion
    try:
        from src.dwh.etl_pipeline import record_screening_event
        record_screening_event(
            image_bytes=raw_bytes,
            image_dimensions=pil_img.size,
            quality_result=qc_result,
            ita_deg=ita_deg,
            tone_band=tone_band,
            fst=fst,
            predicted_disease=display_disease,
            confidence_pct=round(confidence * 100.0, 1),
            entropy=round(entropy, 3),
            uncertainty_level=uncertainty_level,
            acne_severity=acne_severity,
            observable_features=explanation["observable_features"],
            clinical_rationale=explanation["clinical_rationale"]
        )
    except Exception as dwh_err:
        print(f"[DWH Warning] Streaming logging notice: {dwh_err}")

    return {
        "success": True,
        "predicted_disease": display_disease,
        "confidence_pct": round(confidence * 100.0, 1),
        "entropy": round(entropy, 3),
        "uncertainty_level": uncertainty_level,
        "acne_severity": acne_severity,
        "fitzpatrick_scale": fst,
        "ita_degrees": ita_deg,
        "ita_tone_band": tone_band,
        "quality_warning": quality_warning,
        "observable_features": explanation["observable_features"],
        "clinical_rationale": explanation["clinical_rationale"],
        "referral_guidance": explanation["referral_guidance"],
        "original_b64": pil_to_base64(pil_img.resize((320, 320))),
        "gradcam_b64": pil_to_base64(overlay_img.resize((320, 320)))
    }


@app.route("/api/screen", methods=["POST"])
def screen():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        raw_bytes = file.read()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        res = _run_clinical_screening(pil_img, raw_bytes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference exception: {str(e)}"}), 500


@app.route("/api/screen_base64", methods=["POST"])
def screen_base64():
    """Mobile endpoint for Flutter applications sending base64-encoded images."""
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"success": False, "error": "Missing 'image_base64' in JSON body"}), 400

    try:
        b64_str = data["image_base64"]
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        raw_bytes = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        res = _run_clinical_screening(pil_img, raw_bytes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": f"Base64 decoding/inference error: {str(e)}"}), 500


@app.route("/api/dwh/analytics", methods=["GET"])
def dwh_analytics():
    """Analytics endpoint querying the SQLite Data Warehouse Data Marts for mobile dashboards."""
    try:
        import sqlite3
        dwh_path = os.path.join("data", "warehouse", "fairskin_dwh.db")
        if not os.path.exists(dwh_path):
            return jsonify({"status": "no_dwh_available"}), 404

        conn = sqlite3.connect(dwh_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM fact_screenings")
        total_screenings = cur.fetchone()[0]

        cur.execute("SELECT fitzpatrick_scale, ita_tone_band, total_screenings, mean_confidence FROM mart_fairness_audit")
        fairness_mart = [
            {"fst": r[0], "tone": r[1], "count": r[2], "avg_confidence": r[3]}
            for r in cur.fetchall()
        ]

        cur.execute("SELECT disease_name, icd10_code, screening_count, avg_confidence FROM mart_disease_prevalence")
        prevalence_mart = [
            {"disease": r[0], "icd10": r[1], "count": r[2], "avg_confidence": r[3]}
            for r in cur.fetchall()
        ]
        conn.close()

        return jsonify({
            "status": "success",
            "total_screenings": total_screenings,
            "fairness_audit": fairness_mart,
            "disease_prevalence": prevalence_mart
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("FairSkin AI Screening Engine Running Locally (100% Offline)")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
