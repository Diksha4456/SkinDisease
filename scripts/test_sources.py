import urllib.request
import re

url = 'http://www.atlasdermatologico.com.br/browse.jsf'
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')
    targets = {
        'acne': 3,
        'melasma': 285,
        'psoriasis': 395,
        'tinea_corporis': 454,
        'vitiligo': 487
    }
    for disease_name, did in targets.items():
        d_url = f'http://www.atlasdermatologico.com.br/disease.jsf?diseaseId={did}'
        req = urllib.request.Request(d_url, headers={'User-Agent': 'Mozilla/5.0'})
        d_html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')
        img_ids = re.findall(r'imageId=(\d+)', d_html)
        unique_imgs = sorted(set(img_ids), key=int)
        print(f"Disease: {disease_name:<15} (ID {did}) -> {len(unique_imgs)} images: {unique_imgs[:5]}...")







except Exception as e:
    print('Err:', e)
