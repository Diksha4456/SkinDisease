"""
FairSkin AI — Data Warehouse ETL Pipeline.
Extracts, transforms, and loads clinical screening events and training records
into the Star Schema database.
"""

import os
import sys
import hashlib
import uuid
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dwh.warehouse_schema import get_db_connection, initialize_warehouse, DWH_PATH


def record_screening_event(
    image_bytes: bytes,
    image_dimensions: tuple,
    quality_result: Dict[str, Any],
    ita_deg: float,
    tone_band: str,
    fst: int,
    predicted_disease: str,
    confidence_pct: float,
    entropy: float,
    uncertainty_level: str,
    acne_severity: str,
    observable_features: List[str],
    clinical_rationale: str
) -> str:
    """
    Real-time streaming ETL: Writes a screening event directly into the Star Schema DWH.
    """
    screening_uuid = str(uuid.uuid4())
    img_sha256 = hashlib.sha256(image_bytes).hexdigest()
    w, h = image_dimensions
    aspect_ratio = round(w / max(h, 1), 3)

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1. Dimension: Image Metadata (Insert or find existing)
        quality_status = "PASSED" if quality_result.get("passed", True) else "FLAGGED"
        quality_warning = "; ".join(quality_result.get("flags", [])) if quality_result.get("flags") else None

        cur.execute("""
        INSERT OR IGNORE INTO dim_image_metadata (image_sha256, width_px, height_px, aspect_ratio, quality_status, quality_warning, source_origin)
        VALUES (?, ?, ?, ?, ?, ?, 'LiveScreening')
        """, (img_sha256, w, h, aspect_ratio, quality_status, quality_warning))

        cur.execute("SELECT image_key FROM dim_image_metadata WHERE image_sha256 = ?", (img_sha256,))
        row = cur.fetchone()
        image_key = row["image_key"] if row else 1

        # 2. Dimension: Patient Skin Tone
        melanin_group = "High" if fst >= 5 else ("Medium" if fst in [3, 4] else "Low")
        cur.execute("""
        INSERT INTO dim_patient_skin (ita_degrees, ita_tone_band, fitzpatrick_scale, melanin_risk_group)
        VALUES (?, ?, ?, ?)
        """, (ita_deg, tone_band, fst, melanin_group))
        skin_tone_key = cur.lastrowid

        # 3. Dimension: Disease mapping
        disease_lookup = predicted_disease.lower().replace(" ", "_")
        if "acne" in disease_lookup:
            d_code = "acne"
        elif "melasma" in disease_lookup or "chloasma" in disease_lookup:
            d_code = "melasma"
        elif "psoriasis" in disease_lookup:
            d_code = "psoriasis"
        elif "ringworm" in disease_lookup or "tinea" in disease_lookup:
            d_code = "ringworm"
        elif "vitiligo" in disease_lookup:
            d_code = "vitiligo"
        else:
            d_code = "non_target_ood"

        cur.execute("SELECT disease_key FROM dim_disease WHERE disease_code = ?", (d_code,))
        row = cur.fetchone()
        predicted_disease_key = row["disease_key"] if row else 6

        # 4. Model Key
        cur.execute("SELECT model_key FROM dim_model_version WHERE is_active = 1 ORDER BY model_key DESC LIMIT 1")
        row = cur.fetchone()
        model_key = row["model_key"] if row else 1

        # 5. Fact: fact_screenings
        referral_rec = (confidence_pct < 40.0) or (d_code == "non_target_ood")
        features_json = json.dumps(observable_features)

        cur.execute("""
        INSERT INTO fact_screenings (
            screening_uuid, image_key, skin_tone_key, predicted_disease_key, model_key,
            confidence_pct, prediction_entropy, uncertainty_level, acne_severity,
            clinical_rationale, top_detected_features, referral_recommended
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            screening_uuid, image_key, skin_tone_key, predicted_disease_key, model_key,
            confidence_pct, entropy, uncertainty_level, acne_severity,
            clinical_rationale, features_json, 1 if referral_rec else 0
        ))

        conn.commit()
    except Exception as e:
        print(f"[DWH ETL Error] Failed recording screening: {e}")
    finally:
        conn.close()

    return screening_uuid


def initial_warehouse_load_from_manifest(manifest_csv: str = "data/manifests/master_clinical_manifest.csv"):
    """
    Batch ETL: Ingests all historical training, validation, and test dataset records into DWH.
    """
    if not os.path.exists(manifest_csv):
        print(f"[DWH ETL] Manifest {manifest_csv} not found, skipping batch load.")
        return

    initialize_warehouse()
    df = pd.read_csv(manifest_csv)
    print(f"[DWH Batch ETL] Ingesting {len(df)} historical dataset records into Star Schema...")

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT model_key FROM dim_model_version WHERE is_active = 1 LIMIT 1")
    model_key = cur.fetchone()["model_key"]

    inserted = 0
    for _, row in df.iterrows():
        try:
            sha = str(row.get("sha256", hashlib.sha256(str(row["file_path"]).encode()).hexdigest()))
            w = int(row.get("image_width", 224))
            h = int(row.get("image_height", 224))
            ar = round(w / max(h, 1), 3)
            q_status = str(row.get("quality_status", "PASSED"))
            q_flags = str(row.get("quality_flags", ""))
            src = str(row.get("source_dataset", "Historical"))

            cur.execute("""
            INSERT OR IGNORE INTO dim_image_metadata (image_sha256, width_px, height_px, aspect_ratio, quality_status, quality_warning, source_origin)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sha, w, h, ar, q_status, q_flags if q_flags != "none" else None, src))

            cur.execute("SELECT image_key FROM dim_image_metadata WHERE image_sha256 = ?", (sha,))
            img_key = cur.fetchone()["image_key"]

            ita = float(row.get("ita_degrees", 30.0))
            band = str(row.get("ita_tone_band", "Intermediate"))
            fst = int(row.get("fitzpatrick_scale", 3))
            m_grp = "High" if fst >= 5 else ("Medium" if fst in [3, 4] else "Low")

            cur.execute("""
            INSERT INTO dim_patient_skin (ita_degrees, ita_tone_band, fitzpatrick_scale, melanin_risk_group)
            VALUES (?, ?, ?, ?)
            """, (ita, band, fst, m_grp))
            skin_key = cur.lastrowid

            d_code = str(row.get("disease_label", "non_target_ood"))
            cur.execute("SELECT disease_key FROM dim_disease WHERE disease_code = ?", (d_code,))
            d_res = cur.fetchone()
            d_key = d_res["disease_key"] if d_res else 6

            acne_sev = str(row.get("acne_severity", "N/A"))
            feat_json = str(row.get("observable_features", "[]"))
            s_uuid = f"hist_{row.get('image_id', uuid.uuid4())}"

            cur.execute("""
            INSERT OR IGNORE INTO fact_screenings (
                screening_uuid, image_key, skin_tone_key, predicted_disease_key, model_key,
                confidence_pct, prediction_entropy, uncertainty_level, acne_severity,
                clinical_rationale, top_detected_features, referral_recommended
            )
            VALUES (?, ?, ?, ?, ?, 95.0, 0.15, 'Low', ?, 'Verified historical clinical record.', ?, 0)
            """, (s_uuid, img_key, skin_key, d_key, model_key, acne_sev, feat_json))

            inserted += 1
        except Exception as e:
            continue

    conn.commit()
    conn.close()
    print(f"[DWH Batch ETL] Successfully populated DWH with {inserted} historical clinical facts.")


if __name__ == "__main__":
    initial_warehouse_load_from_manifest()
