"""
FairSkin AI — Data Warehouse Analytics & OLAP Query Suite.
Executes business intelligence and fairness queries across the dimensional marts.
"""

import os
import sys
import pandas as pd
import sqlite3

DWH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "warehouse", "fairskin_dwh.db"))


def run_warehouse_analytics():
    if not os.path.exists(DWH_PATH):
        print(f"Data Warehouse database not found at {DWH_PATH}")
        return

    conn = sqlite3.connect(DWH_PATH)

    print("=" * 75)
    print("FAIRSKIN AI — DATA WAREHOUSE (DWH) EXECUTIVE ANALYTICS REPORT")
    print("=" * 75)

    # 1. Total Volume in Fact Table
    total_screenings = conn.execute("SELECT COUNT(*) FROM fact_screenings").fetchone()[0]
    total_patients = conn.execute("SELECT COUNT(*) FROM dim_patient_skin").fetchone()[0]
    total_images = conn.execute("SELECT COUNT(*) FROM dim_image_metadata").fetchone()[0]
    print(f"\n[DWH Summary] Fact Screenings: {total_screenings} | Registered Patients: {total_patients} | Stored Images: {total_images}")

    # 2. Mart: Fairness Audit by Fitzpatrick Scale
    print("\n" + "-" * 75)
    print("DATA MART: FAIRNESS AUDIT BY FITZPATRICK SKIN TONE (FST I - VI)")
    print("-" * 75)
    df_fairness = pd.read_sql_query("SELECT * FROM mart_fairness_audit", conn)
    print(df_fairness.to_string(index=False))

    # 3. Mart: Disease Prevalence & Average Confidence
    print("\n" + "-" * 75)
    print("DATA MART: CLINICAL DISEASE PREVALENCE & ACCURACY PROFILE")
    print("-" * 75)
    df_prev = pd.read_sql_query("SELECT * FROM mart_disease_prevalence", conn)
    print(df_prev.to_string(index=False))

    # 4. Mart: Quality Monitoring
    print("\n" + "-" * 75)
    print("DATA MART: IMAGE QUALITY & TRIAGE METRICS")
    print("-" * 75)
    df_qual = pd.read_sql_query("SELECT * FROM mart_quality_monitoring", conn)
    print(df_qual.to_string(index=False))

    print("\n" + "=" * 75)
    print(f"Data Warehouse file verified at: {DWH_PATH}")
    print("=" * 75)
    conn.close()


if __name__ == "__main__":
    run_warehouse_analytics()
