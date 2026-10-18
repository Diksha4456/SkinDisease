"""
Download only explicitly approved image URLs.
Input CSV columns:
  image_id,url,source_dataset,license_status,permission_reference
Only rows with license_status exactly 'verified' are attempted.
This is a safety/provenance guard, not legal advice.
"""
import argparse, os, re, hashlib, time
import pandas as pd
import requests
from urllib.parse import urlparse

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+","_",str(value))[:100]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out",default="data/raw")
    ap.add_argument("--timeout",type=int,default=20)
    ap.add_argument("--pause",type=float,default=0.5)
    args=ap.parse_args()
    df=pd.read_csv(args.manifest)
    needed={"image_id","url","source_dataset","license_status","permission_reference"}
    missing=needed-set(df.columns)
    if missing: raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    os.makedirs(args.out,exist_ok=True)
    log=[]
    for row in df.to_dict("records"):
        if str(row["license_status"]).strip().lower()!="verified":
            log.append({"image_id":row["image_id"],"status":"skipped_unverified_permission","path":""})
            continue
        url=str(row["url"])
        parsed=urlparse(url)
        if parsed.scheme not in ("http","https") or not parsed.netloc:
            log.append({"image_id":row["image_id"],"status":"invalid_url","path":""}); continue
        try:
            r=requests.get(url,timeout=args.timeout,headers={"User-Agent":"FairSkinAI-Research/1.0"})
            r.raise_for_status()
            ctype=r.headers.get("Content-Type","").lower()
            if not ctype.startswith("image/"):
                log.append({"image_id":row["image_id"],"status":"not_image_content","path":""}); continue
            digest=hashlib.sha256(r.content).hexdigest()
            ext={"image/jpeg":".jpg","image/png":".png","image/webp":".webp"}.get(ctype.split(";")[0],".img")
            name=safe_name(row["image_id"])+"_"+digest[:12]+ext
            path=os.path.join(args.out,name)
            with open(path,"wb") as f: f.write(r.content)
            log.append({"image_id":row["image_id"],"status":"downloaded","path":path,"sha256":digest,"source_dataset":row["source_dataset"],"permission_reference":row["permission_reference"]})
        except Exception as e:
            log.append({"image_id":row["image_id"],"status":"download_error:"+type(e).__name__,"path":""})
        time.sleep(args.pause)
    pd.DataFrame(log).to_csv(os.path.join(args.out,"download_log.csv"),index=False)
    print("Finished. See",os.path.join(args.out,"download_log.csv"))

if __name__=="__main__": main()
