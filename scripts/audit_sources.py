import argparse
import os
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to dataset metadata CSV")
    ap.add_argument("--out", default="reports/source_audit")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df=pd.read_csv(args.csv)
    required={"label","fitzpatrick_scale"}
    missing=required-set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["label_norm"]=df["label"].astype(str).str.strip().str.lower()
    df["skin_type"]=df["fitzpatrick_scale"].map({1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",-1:"Unknown"}).fillna("Missing/other")
    df["url_present"]=df["url"].notna() if "url" in df.columns else False
    df["hash_present"]=df["md5hash"].notna() if "md5hash" in df.columns else False
    pd.crosstab(df["label"],df["skin_type"]).to_csv(os.path.join(args.out,"label_by_skin_type.csv"))
    df.groupby(["label","skin_type"],dropna=False).size().rename("records").reset_index().to_csv(os.path.join(args.out,"label_by_skin_type_long.csv"),index=False)
    if "url" in df.columns:
        df["url_domain"]=df["url"].fillna("").map(lambda u: u.split("/")[2] if u.startswith(("http://","https://")) and len(u.split("/"))>2 else "")
        df["url_domain"].value_counts().rename_axis("domain").reset_index(name="records").to_csv(os.path.join(args.out,"url_domains.csv"),index=False)
    summary={
        "rows":len(df),
        "unique_labels":int(df["label"].nunique()),
        "missing_url":int(df["url"].isna().sum()) if "url" in df.columns else None,
        "duplicate_hash_rows":int(df["md5hash"].duplicated().sum()) if "md5hash" in df.columns else None,
        "duplicate_url_rows":int(df["url"].dropna().duplicated().sum()) if "url" in df.columns else None,
        "unknown_skin_type":int((df["fitzpatrick_scale"]==-1).sum()),
        "missing_skin_type":int(df["fitzpatrick_scale"].isna().sum()),
    }
    pd.Series(summary).to_json(os.path.join(args.out,"summary.json"),indent=2)
    print("Wrote audit outputs to",args.out)
    print(pd.Series(summary).to_string())

if __name__=="__main__": main()
