"""
FairSkin AI — Clinical Screening Data Warehouse (DWH) Architecture.
Star Schema Dimensional Model:
  - Fact: fact_screenings
  - Dimensions: dim_patient_skin, dim_disease, dim_clinical_features, dim_image_metadata, dim_model_version
  - Data Marts: mart_fairness_audit, mart_disease_prevalence, mart_quality_monitoring
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional

DWH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "warehouse"))
DWH_PATH = os.path.join(DWH_DIR, "fairskin_dwh.db")


def get_db_connection():
    os.makedirs(DWH_DIR, exist_ok=True)
    conn = sqlite3.connect(DWH_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_warehouse():
    """Creates the Star Schema tables and Data Mart views."""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Dimension: Disease Taxonomy (ICD-10 clinical mapping)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_disease (
        disease_key INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_code TEXT UNIQUE NOT NULL,
        disease_name TEXT NOT NULL,
        icd10_code TEXT NOT NULL,
        clinical_category TEXT NOT NULL,
        urgency_level TEXT NOT NULL,
        referral_protocol TEXT NOT NULL
    )
    """)

    # Populate static disease dimensions
    disease_seed = [
        ("acne", "Acne Vulgaris", "L70.0", "Sebaceous Follicular", "Routine", "Topical retinoids/antimicrobial assessment"),
        ("melasma", "Melasma / Chloasma", "L81.1", "Pigmentary Dyschromia", "Routine", "Photoprotection & tyrosinase inhibitor therapy"),
        ("psoriasis", "Plaque Psoriasis", "L40.0", "Papulosquamous Autoimmune", "Moderate", "Keratolytic agents, topical corticosteroids, phototherapy"),
        ("ringworm", "Tinea Corporis / Ringworm", "B35.4", "Fungal Dermatophytosis", "Routine", "Topical/oral antifungal confirmation via KOH mount"),
        ("vitiligo", "Vitiligo", "L80", "Autoimmune Depigmentation", "Routine", "Topical calcineurin inhibitors, narrowband UVB referral"),
        ("non_target_ood", "Condition Not Recognized", "R21", "Atypical / Non-Target Cutaneous", "Prompt Clinical Review", "Formal in-person dermatological evaluation recommended")
    ]
    cur.executemany("""
    INSERT OR IGNORE INTO dim_disease (disease_code, disease_name, icd10_code, clinical_category, urgency_level, referral_protocol)
    VALUES (?, ?, ?, ?, ?, ?)
    """, disease_seed)

    # 2. Dimension: Patient Skin Tone (ITA & Fitzpatrick)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_patient_skin (
        skin_tone_key INTEGER PRIMARY KEY AUTOINCREMENT,
        ita_degrees REAL NOT NULL,
        ita_tone_band TEXT NOT NULL,
        fitzpatrick_scale INTEGER NOT NULL,
        melanin_risk_group TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Dimension: Image Quality & Acquisition Metadata
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_image_metadata (
        image_key INTEGER PRIMARY KEY AUTOINCREMENT,
        image_sha256 TEXT UNIQUE NOT NULL,
        width_px INTEGER NOT NULL,
        height_px INTEGER NOT NULL,
        aspect_ratio REAL NOT NULL,
        quality_status TEXT NOT NULL,
        quality_warning TEXT,
        source_origin TEXT DEFAULT 'WebUpload',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 4. Dimension: Deep Learning Model Version
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_model_version (
        model_key INTEGER PRIMARY KEY AUTOINCREMENT,
        version_tag TEXT UNIQUE NOT NULL,
        backbone_architecture TEXT NOT NULL,
        train_macro_f1 REAL NOT NULL,
        test_accuracy REAL NOT NULL,
        device_backend TEXT NOT NULL,
        is_active BOOLEAN DEFAULT 1
    )
    """)

    cur.execute("""
    INSERT OR IGNORE INTO dim_model_version (version_tag, backbone_architecture, train_macro_f1, test_accuracy, device_backend, is_active)
    VALUES ('v2.3-fairskin-multitask', 'MobileNetV3-Small (Multi-Task Feature Head)', 0.9723, 0.9630, 'CPU-Local', 1)
    """)

    # 5. Dimension: Observable Visual Features
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_clinical_features (
        feature_key INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT UNIQUE NOT NULL,
        target_disease TEXT NOT NULL,
        clinical_description TEXT NOT NULL
    )
    """)

    features_seed = [
        ("erythematous_papule_pustule", "acne", "Discrete elevated inflammatory papules with or without purulence"),
        ("open_closed_comedo", "acne", "Follicular impactions (blackheads and whiteheads)"),
        ("inflammatory_nodule_scarring", "acne", "Deep inflammatory nodules, cysts, or cicatricial changes"),
        ("reticulated_hyperpigmentation", "melasma", "Network-like brown macular pigmentation across sun-exposed areas"),
        ("irregular_pigment_border", "melasma", "Geographic or serrated hypermelanotic borders"),
        ("chalky_white_depigmentation", "vitiligo", "Complete loss of epidermal melanin with porcelain appearance"),
        ("well_demarcated_border", "vitiligo", "Sharply defined borders between normal and depigmented skin"),
        ("erythematous_scaly_plaque", "psoriasis", "Well-circumscribed salmon-pink inflammatory plaques"),
        ("micaceous_silvery_scale", "psoriasis", "Characteristic multi-layered silvery scales (Auspitz suspect)"),
        ("annular_erythematous_ring", "ringworm", "Circular or polycyclic erythematous active expanding border"),
        ("active_scaly_advancing_border", "ringworm", "Elevated leading edge with fine scale"),
        ("central_clearing", "ringworm", "Resolution of inflammation in the center of the ring")
    ]
    cur.executemany("""
    INSERT OR IGNORE INTO dim_clinical_features (feature_name, target_disease, clinical_description)
    VALUES (?, ?, ?)
    """, features_seed)

    # 6. Fact Table: Clinical Screenings (Transaction Fact)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_screenings (
        screening_id INTEGER PRIMARY KEY AUTOINCREMENT,
        screening_uuid TEXT UNIQUE NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_key INTEGER NOT NULL,
        skin_tone_key INTEGER NOT NULL,
        predicted_disease_key INTEGER NOT NULL,
        model_key INTEGER NOT NULL,
        confidence_pct REAL NOT NULL,
        prediction_entropy REAL NOT NULL,
        uncertainty_level TEXT NOT NULL,
        acne_severity TEXT DEFAULT 'N/A',
        clinical_rationale TEXT,
        top_detected_features TEXT,
        referral_recommended BOOLEAN DEFAULT 0,
        FOREIGN KEY (image_key) REFERENCES dim_image_metadata(image_key),
        FOREIGN KEY (skin_tone_key) REFERENCES dim_patient_skin(skin_tone_key),
        FOREIGN KEY (predicted_disease_key) REFERENCES dim_disease(disease_key),
        FOREIGN KEY (model_key) REFERENCES dim_model_version(model_key)
    )
    """)

    # 7. Analytical Data Marts (SQL Views)
    cur.execute("""
    CREATE VIEW IF NOT EXISTS mart_fairness_audit AS
    SELECT 
        s.fitzpatrick_scale,
        s.ita_tone_band,
        COUNT(f.screening_id) AS total_screenings,
        ROUND(AVG(f.confidence_pct), 2) AS mean_confidence,
        ROUND(AVG(f.prediction_entropy), 3) AS mean_entropy,
        SUM(CASE WHEN f.uncertainty_level = 'High' THEN 1 ELSE 0 END) AS high_uncertainty_count,
        SUM(CASE WHEN f.referral_recommended = 1 THEN 1 ELSE 0 END) AS referrals_count
    FROM fact_screenings f
    JOIN dim_patient_skin s ON f.skin_tone_key = s.skin_tone_key
    GROUP BY s.fitzpatrick_scale, s.ita_tone_band
    ORDER BY s.fitzpatrick_scale
    """)

    cur.execute("""
    CREATE VIEW IF NOT EXISTS mart_disease_prevalence AS
    SELECT 
        d.disease_name,
        d.icd10_code,
        COUNT(f.screening_id) AS screening_count,
        ROUND(AVG(f.confidence_pct), 2) AS avg_confidence,
        SUM(CASE WHEN f.acne_severity = 'High' THEN 1 ELSE 0 END) AS severe_acne_count,
        SUM(CASE WHEN f.acne_severity = 'Low' THEN 1 ELSE 0 END) AS mild_acne_count
    FROM fact_screenings f
    JOIN dim_disease d ON f.predicted_disease_key = d.disease_key
    GROUP BY d.disease_name, d.icd10_code
    ORDER BY screening_count DESC
    """)

    cur.execute("""
    CREATE VIEW IF NOT EXISTS mart_quality_monitoring AS
    SELECT 
        m.quality_status,
        m.quality_warning,
        COUNT(f.screening_id) AS total_cases,
        ROUND(AVG(f.confidence_pct), 2) AS avg_confidence
    FROM fact_screenings f
    JOIN dim_image_metadata m ON f.image_key = m.image_key
    GROUP BY m.quality_status, m.quality_warning
    """)

    conn.commit()
    conn.close()
    print(f"[DWH] Initialized FairSkin Star Schema Data Warehouse at: {DWH_PATH}")


if __name__ == "__main__":
    initialize_warehouse()
